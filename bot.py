#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
M3U Smart Toolkit — Telegram Bot
Author: You & ChatGPT
Strings: English-only
OS: Cross-platform (Linux/Mac/Windows). Network checks best on Linux/Mac.

Features
--------
• Upload one/many .m3u/.m3u8 files
• Merge playlists (order-preserving), de-duplicate by (Channel Name, URL)
• Channel Status Analyzer with emoji:
    🟢 Active (OK & playable)
    🟡 Incomplete (no valid stream URL)
    🔴 Not working (bad status / timeout / network error)
    🔵 200 OK but Not Media (HTML/JSON/text, not audio/video/HLS)
• Reformatter:
    #Channel → 001
    #EXTINF:-1 tvg-logo="..." , Channel Name
    http(s)://stream-url
• Remove Categories (drop group-title attr from EXTINF)
• Bulk comment/uncomment by name filter
• Compare two playlists by Channel Name: report names present in B but not A
• Export:
    - Full analyzed report (emoji + lines)
    - Fresh Active-only playlist (numbered)
    - Merged & reformatted playlist
"""

import os
import re
import sys
import json
import csv
import asyncio
import aiohttp
import logging
import io
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any

from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ConversationHandler,
    ContextTypes,
)
from telegram.constants import ParseMode

# --- Setup Logging ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# =============================================================================
#
# BEGIN: COPIED LOGIC FROM CLI SCRIPT
# (Mostly unchanged, except for parse_m3u and write_playlist)
#
# =============================================================================

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico", ".bmp")
VIDEO_AUDIO_CT_PREFIX = ("video/", "audio/")
HLS_CTS = (
    "application/vnd.apple.mpegurl",
    "application/x-mpegurl",
    "audio/mpegurl",
    "application/mpegurl",
)
TEXTY_CTS = ("text/plain", "text/html", "application/json", "application/xml")
OK_STATUSES = (200, 206)

# ---------------------- Utils ---------------------- #
def is_http_url(s: str) -> bool:
    s = s.strip().lower()
    return s.startswith("http://") or s.startswith("https://")

def is_commented(s: str) -> bool:
    return s.strip().startswith("#")

def looks_like_image_url(url: str) -> bool:
    u = url.split("?", 1)[0].split("#", 1)[0].strip().lower()
    return u.endswith(IMAGE_EXTS)

def first_valid_stream(urls: List[str]) -> Optional[str]:
    for u in urls:
        if is_http_url(u) and (not is_commented(u)) and (not looks_like_image_url(u)):
            return u.strip()
    return None

def extract_attr(extinf: str, attr: str) -> Optional[str]:
    # e.g. attr = tvg-logo, group-title
    # pattern like tvg-logo="..."; group-title="..."
    m = re.search(rf'{re.escape(attr)}="([^"]*)"', extinf, flags=re.IGNORECASE)
    return m.group(1).strip() if m else None

def extract_name_from_extinf(extinf: str) -> str:
    # EXTINF:-1 ... ,Channel Name
    p = extinf.rsplit(",", 1)
    if len(p) == 2:
        return p[1].strip()
    return "Unknown"

# ---------------------- Parser ---------------------- #
def parse_m3u_from_content(filename: str, content: str) -> List[Dict[str, Any]]:
    """
    MODIFIED: Parses M3U content from a string instead of a file path.
    
    Returns a list of entries:
    {
        "source": filename,
        "extinf": "#EXTINF:-1 tvg-logo=\"...\" group-title=\"...\",Name",
        "name": "Name",
        "logo": "...",
        "group": "...",
        "urls": [url1, url2, ...]  # within this block
    }
    """
    entries: List[Dict[str, Any]] = []
    try:
        lines = [ln.rstrip("\n") for ln in content.splitlines()]
    except Exception as e:
        logger.warning(f"Could not read content for {filename}: {e}")
        return entries

    current_extinf = None
    current_urls: List[str] = []

    def flush_block():
        nonlocal current_extinf, current_urls
        if current_extinf:
            name = extract_name_from_extinf(current_extinf)
            entries.append({
                "source": filename, # Use filename passed to function
                "extinf": current_extinf,
                "name": name,
                "logo": extract_attr(current_extinf, "tvg-logo"),
                "group": extract_attr(current_extinf, "group-title"),
                "urls": [u for u in current_urls if u.strip()],
            })
        current_extinf = None
        current_urls = []

    for raw in lines:
        line = raw.strip()
        if line.startswith("#EXTINF"):
            flush_block()
            current_extinf = line
        else:
            if current_extinf:
                # Collect candidate URLs / lines inside block
                if is_http_url(line) or line.startswith("#"):
                    current_urls.append(line)

    flush_block()
    return entries

# ---------------------- Merge & Format ---------------------- #
def merge_entries(list_of_lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen = set()
    for entries in list_of_lists:
        for e in entries:
            # dedupe by (name, first_valid_stream) pair
            url = first_valid_stream(e["urls"]) or ""
            key = (e["name"].strip().lower(), url.strip().lower())
            if key in seen:
                continue
            seen.add(key)
            merged.append(e)
    return merged

def drop_group_title(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for e in entries:
        if 'group-title="' in e["extinf"]:
            new_extinf = re.sub(r'\s*group-title="[^"]*"', "", e["extinf"], flags=re.IGNORECASE)
        else:
            new_extinf = e["extinf"]
        # Also clean extra spaces before comma if needed
        new_extinf = re.sub(r'\s+,', ' ,', new_extinf)
        ee = dict(e)
        ee["extinf"] = new_extinf.strip()
        out.append(ee)
    return out

def render_numbered_playlist(entries: List[Dict[str, Any]], only_active: Optional[Dict[str, str]] = None) -> str:
    """
    only_active: mapping name->status to include only 🟢 channels (if provided)
    """
    out = ["#EXTM3U\n"]
    numbered = []
    for e in entries:
        if only_active is not None:
            st = only_active.get(e["name"], "")
            if st != "🟢":
                continue
        url = first_valid_stream(e["urls"])
        if not url:
            continue
        numbered.append((e["extinf"], url))
    width = max(3, len(str(len(numbered))))
    for idx, (extinf, url) in enumerate(numbered, start=1):
        out.append(f"#Channel → {idx:0{width}d}\n")
        out.append(extinf if extinf.endswith("\n") else (extinf + "\n"))
        out.append(url if url.endswith("\n") else (url + "\n"))
        out.append("\n")
    return "".join(out)

def render_analyzed_report(entries: List[Dict[str, Any]], status_map: Dict[str, str]) -> str:
    """
    Build emoji report preserving original EXTINF/URL pair (first valid URL).
    """
    out = ["#EXTM3U\n"]
    for e in entries:
        url = first_valid_stream(e["urls"])
        if not url:
            emoji = "🟡"
            out.append(f"{emoji} {e['extinf']}\n")
            out.append("\n")
            continue
        emoji = status_map.get(e["name"], "🔴")
        out.append(f"{emoji} {e['extinf']}\n")
        out.append(url + ("\n" if not url.endswith("\n") else ""))
        out.append("\n")
    return "".join(out)

# ---------------------- Analyzer (Async) ---------------------- #
async def sniff_url(session: aiohttp.ClientSession, url: str, timeout_s: float = 7.5) -> Tuple[str, str]:
    """
    Returns (emoji_status, detail_reason)
    🟢 playable; 🔵 not-media; 🔴 bad; (🟡 handled at caller when url missing)
    """
    try:
        # HEAD first (some servers block HEAD; we fallback GET with small range)
        try:
            async with session.head(url, timeout=timeout_s) as resp:
                status = resp.status
                ctype = resp.headers.get("Content-Type", "").lower()
                if status in OK_STATUSES:
                    # Evaluate by content-type
                    if ctype.startswith(VIDEO_AUDIO_CT_PREFIX) or ctype in HLS_CTS:
                        return "🟢", f"OK {status} {ctype or ''}".strip()
                    if any(ctype.startswith(t) for t in TEXTY_CTS):
                        return "🔵", f"OK {status} but text-like {ctype}"
                    # Unknown ctype — try a tiny GET range
                else:
                    return "🔴", f"HTTP {status}"
        except Exception:
            # fall-through to GET
            pass

        headers = {"Range": "bytes=0-1023"}
        async with session.get(url, headers=headers, timeout=timeout_s) as resp:
            status = resp.status
            ctype = resp.headers.get("Content-Type", "").lower()
            content = await resp.content.read(1024)
            if status in OK_STATUSES:
                # Heuristics:
                if ctype.startswith(VIDEO_AUDIO_CT_PREFIX) or ctype in HLS_CTS:
                    return "🟢", f"OK {status} {ctype or ''}".strip()
                # HLS by content sniff: #EXTM3U
                sniff = content[:256].decode("latin-1", errors="ignore")
                if "#EXTM3U" in sniff:
                    return "🟢", f"OK {status} HLS playlist"
                if any(ctype.startswith(t) for t in TEXTY_CTS) or ("<html" in sniff.lower()) or ("{" in sniff[:5]):
                    return "🔵", f"OK {status} but not media"
                # Unknown but OK: treat as playable fallback
                return "🟢", f"OK {status} unknown-media"
            else:
                return "🔴", f"HTTP {status}"
    except asyncio.TimeoutError:
        return "🔴", "timeout"
    except Exception as ex:
        return "🔴", f"error {type(ex).__name__}"

async def analyze_entries(entries: List[Dict[str, Any]], max_conn: int = 32) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Returns:
      status_map: name -> emoji
      reason_map: name -> reason string
    """
    connector = aiohttp.TCPConnector(limit=max_conn, ssl=False)
    timeout = aiohttp.ClientTimeout(total=10)
    status_map: Dict[str, str] = {}
    reason_map: Dict[str, str] = {}

    tasks = []
    names_urls: List[Tuple[str, Optional[str]]] = []
    for e in entries:
        url = first_valid_stream(e["urls"])
        names_urls.append((e["name"], url))

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        sem = asyncio.Semaphore(max_conn)

        async def worker(name: str, url: Optional[str]):
            if not url:
                status_map[name] = "🟡"
                reason_map[name] = "No stream URL"
                return
            if looks_like_image_url(url):
                status_map[name] = "🔵"
                reason_map[name] = "Looks like image URL"
                return
            async with sem:
                emoji, reason = await sniff_url(session, url)
                status_map[name] = emoji
                reason_map[name] = reason

        # NOTE: tqdm progress bar is removed for the bot
        for name, url in names_urls:
            task = asyncio.create_task(worker(name, url))
            tasks.append(task)
        await asyncio.gather(*tasks, return_exceptions=True)

    return status_map, reason_map

# ---------------------- Compare ---------------------- #
def compare_by_name(entries_a: List[Dict[str, Any]], entries_b: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Return channels present in B but not in A (by Name).
    """
    names_a = {e["name"].strip().lower() for e in entries_a}
    uniq_b = []
    seen = set()
    for e in entries_b:
        key = e["name"].strip().lower()
        if key not in names_a and key not in seen:
            uniq_b.append(e)
            seen.add(key)
    return uniq_b

# ---------------------- Comment/Uncomment ---------------------- #
def toggle_comment(entries: List[Dict[str, Any]], contains: str, comment: bool) -> List[Dict[str, Any]]:
    """
    Comment or uncomment blocks whose channel name contains a keyword (case-insensitive).
    Commenting means prefixing stream URL with '#', and also prefixing EXTINF line with '#' (soft disable).
    """
    kw = contains.lower()
    out = []
    for e in entries:
        if kw in e["name"].lower():
            ee = dict(e)
            if comment:
                ext = ee["extinf"]
                if not ext.lstrip().startswith("#"):
                    ee["extinf"] = "#" + ext
                new_urls = []
                for u in ee["urls"]:
                    if is_http_url(u) and not u.lstrip().startswith("#"):
                        new_urls.append("#" + u)
                    else:
                        new_urls.append(u)
                ee["urls"] = new_urls
            else:
                # uncomment
                ee["extinf"] = ee["extinf"].lstrip("#").strip()
                ee["urls"] = [u.lstrip("#").strip() for u in ee["urls"]]
            out.append(ee)
        else:
            out.append(e)
    return out

# ---------------------- Writer ---------------------- #
def render_clean_playlist(entries: List[Dict[str, Any]]) -> str:
    """
    MODIFIED: Renders a clean M3U playlist to a string.
    """
    # keep first valid url only to stay TV-friendly
    lines = ["#EXTM3U\n"]
    for e in entries:
        url = first_valid_stream(e["urls"])
        if not url:
            continue
        extinf = e["extinf"]
        if not extinf.endswith("\n"):
            extinf += "\n"
        lines.append(extinf)
        lines.append(url if url.endswith("\n") else (url + "\n"))
        lines.append("\n")
    return "".join(lines)

def write_playlist(entries: List[Dict[str, Any]], path: Path):
    """
    ORIGINAL: Kept for compatibility, but bot uses render_clean_playlist.
    """
    content = render_clean_playlist(entries)
    path.write_text(content, encoding="utf-8")

def write_text(text: str, path: Path):
    """
    ORIGINAL: Kept for compatibility, but bot sends text directly.
    """
    path.write_text(text, encoding="utf-8")

# =============================================================================
#
# END: COPIED LOGIC
#
# =============================================================================


# =============================================================================
#
# BEGIN: TELEGRAM BOT LOGIC
#
# =============================================================================

# --- Conversation States ---
(
    SELECTING_FILES,
    MAIN_MENU,
    AWAITING_COMMENT_FILTER,
    AWAITING_UNCOMMENT_FILTER,
    AWAITING_COMPARE_A,
    AWAITING_COMPARE_B,
) = range(6)

# --- Bot Helper Functions ---

def get_main_menu_keyboard() -> ReplyKeyboardMarkup:
    """Returns the main menu keyboard."""
    keyboard = [
        ["1) Analyze channels 🟢🟡🔴🔵"],
        ["2) Reformat numbered playlist"],
        ["3) Merge & export clean"],
        ["4) Remove categories"],
        ["5) Comment by filter"],
        ["6) UN-comment by filter"],
        ["7) Compare (B not in A)"],
        ["8) Export Active-only (needs Analyze)"],
        ["/start (Reset & upload new files)"],
        ["/cancel (Stop)"]
    ]
    return ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)

async def send_file_to_user(update: Update, context: ContextTypes.DEFAULT_TYPE, filename: str, content: str):
    """Encodes string content and sends it as a document."""
    with io.BytesIO(content.encode('utf-8')) as f:
        await context.bot.send_document(
            chat_id=update.effective_chat.id,
            document=f,
            filename=filename
        )
            
async def send_csv_to_user(update: Update, context: ContextTypes.DEFAULT_TYPE, filename: str, data: List[List[str]]):
    """Sends a list of lists as a CSV file."""
    output = io.StringIO()
    writer = csv.writer(output)
    for row in data:
        writer.writerow(row)
    
    output.seek(0)
    with io.BytesIO(output.getvalue().encode('utf-8')) as f:
        await context.bot.send_document(
            chat_id=update.effective_chat.id,
            document=f,
            filename=filename
        )

# --- Conversation Handlers ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Starts the conversation and asks for files."""
    # Reset user data
    context.user_data.clear()
    context.user_data['files'] = {} # Stores {filename: content_string}
    
    await update.message.reply_text(
        "Welcome to the M3U Smart Toolkit Bot! 🤖\n\n"
        "Please upload one or more `.m3u` or `.m3u8` files to get started. "
        "Send /done when you've uploaded all files.",
        reply_markup=ReplyKeyboardRemove()
    )
    return SELECTING_FILES

async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles file uploads."""
    doc = update.message.document
    if not (doc.file_name.lower().endswith(".m3u") or doc.file_name.lower().endswith(".m3u8")):
        await update.message.reply_text("That's not an .m3u or .m3u8 file. Please try again.")
        return SELECTING_FILES
        
    file = await context.bot.get_file(doc.file_id)
    
    # Download file into memory
    with io.BytesIO() as b:
        await file.download_to_memory(b)
        b.seek(0)
        # Decode with the same robust error handling as the original parser
        try:
            content = b.read().decode('utf-8', errors='ignore')
        except Exception as e:
            logger.error(f"Failed to decode file {doc.file_name}: {e}")
            await update.message.reply_text(f"Could not read {doc.file_name}. File skipped.")
            return SELECTING_FILES
            
    context.user_data['files'][doc.file_name] = content
    await update.message.reply_text(
        f"✅ Received <b>{doc.file_name}</b>. "
        f"Total files: {len(context.user_data['files'])}. "
        "Upload more or send /done.",
        parse_mode=ParseMode.HTML
    )
    return SELECTING_FILES

async def done_uploading(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Processes uploaded files and shows the main menu."""
    if not context.user_data.get('files'):
        await update.message.reply_text("You haven't uploaded any files yet. Please upload at least one.")
        return SELECTING_FILES

    await update.message.reply_text(f"Processing {len(context.user_data['files'])} file(s)...")
    
    all_entries_per_file = []
    for name, content in context.user_data['files'].items():
        # Use the modified parser that reads from a string
        all_entries_per_file.append(parse_m3u_from_content(name, content))
        
    context.user_data['merged_entries'] = merge_entries(all_entries_per_file)
    context.user_data['status_map'] = None # Reset analysis
    
    total_channels = len(context.user_data['merged_entries'])
    
    await update.message.reply_text(
        f"All files loaded and merged!\n"
        f"Total unique channels: <b>{total_channels}</b>\n\n"
        "What would you like to do?",
        reply_markup=get_main_menu_keyboard(),
        parse_mode=ParseMode.HTML
    )
    return MAIN_MENU

async def handle_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles main menu selections."""
    choice = update.message.text
    merged_entries = context.user_data.get('merged_entries')
    
    # Safety check
    if merged_entries is None:
         await update.message.reply_text("Error: No data loaded. Please /start over.")
         return await start(update, context) # Restart

    if choice.startswith("1)"):
        return await menu_1_analyze(update, context)
    elif choice.startswith("2)"):
        return await menu_2_reformat(update, context)
    elif choice.startswith("3)"):
        return await menu_3_merge(update, context)
    elif choice.startswith("4)"):
        return await menu_4_remove_categories(update, context)
    elif choice.startswith("5)"):
        await update.message.reply_text("Enter name filter to COMMENT (case-insensitive):", reply_markup=ReplyKeyboardRemove())
        return AWAITING_COMMENT_FILTER
    elif choice.startswith("6)"):
        await update.message.reply_text("Enter name filter to UNCOMMENT (case-insensitive):", reply_markup=ReplyKeyboardRemove())
        return AWAITING_UNCOMMENT_FILTER
    elif choice.startswith("7)"):
        # Start the "Compare" flow
        context.user_data['compare_a_entries'] = None
        await update.message.reply_text("First, please upload the 'A' file (the main/old list).", reply_markup=ReplyKeyboardRemove())
        return AWAITING_COMPARE_A
    elif choice.startswith("8)"):
        return await menu_8_export_active(update, context)
    else:
        await update.message.reply_text("Invalid choice. Please select from the menu.", reply_markup=get_main_menu_keyboard())
        return MAIN_MENU

# --- Menu Action Handlers ---

async def menu_1_analyze(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles Menu 1: Analyze Channels"""
    await update.message.reply_text("Analyzing channels... This may take several minutes. Please wait.", reply_markup=ReplyKeyboardRemove())
    merged_entries = context.user_data['merged_entries']
    
    # Run the original async analyzer function
    status_map, reason_map = await analyze_entries(merged_entries, max_conn=32)
    
    context.user_data['status_map'] = status_map
    context.user_data['reason_map'] = reason_map
    
    await update.message.reply_text("✅ Analysis complete.")
    
    # Send emoji report
    report_txt = render_analyzed_report(merged_entries, status_map)
    await send_file_to_user(update, context, "analyzed_report.m3u", report_txt)
    
    # Send CSV report
    csv_data = [["Channel Name", "Status", "Reason"]]
    for e in merged_entries:
        nm = e["name"]
        st = status_map.get(nm, "🟡")
        rs = reason_map.get(nm, "")
        csv_data.append([nm, st, rs])
    await send_csv_to_user(update, context, "analyzed_reasons.csv", csv_data)

    await update.message.reply_text("What's next?", reply_markup=get_main_menu_keyboard())
    return MAIN_MENU

async def menu_2_reformat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles Menu 2: Reformat Numbered"""
    await update.message.reply_text("Reformatting with numbering...", reply_markup=ReplyKeyboardRemove())
    merged_entries = context.user_data['merged_entries']
    text = render_numbered_playlist(merged_entries, only_active=None)
    await send_file_to_user(update, context, "reformatted_numbered.m3u", text)
    
    await update.message.reply_text("What's next?", reply_markup=get_main_menu_keyboard())
    return MAIN_MENU

async def menu_3_merge(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles Menu 3: Merge & Export"""
    await update.message.reply_text("Exporting merged & clean playlist...", reply_markup=ReplyKeyboardRemove())
    merged_entries = context.user_data['merged_entries']
    
    # Use the new render_clean_playlist function
    content = render_clean_playlist(merged_entries)
    await send_file_to_user(update, context, "merged_clean.m3u", content)
    
    await update.message.reply_text("What's next?", reply_markup=get_main_menu_keyboard())
    return MAIN_MENU

async def menu_4_remove_categories(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles Menu 4: Remove Categories"""
    await update.message.reply_text("Dropping group-title...", reply_markup=ReplyKeyboardRemove())
    merged_entries = context.user_data['merged_entries']
    cleaned = drop_group_title(merged_entries)
    
    content = render_clean_playlist(cleaned)
    await send_file_to_user(update, context, "no_categories.m3u", content)
    
    await update.message.reply_text("What's next?", reply_markup=get_main_menu_keyboard())
    return MAIN_MENU

async def handle_comment_filter(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles Menu 5: Comment by Filter"""
    filt = update.message.text
    if not filt:
        await update.message.reply_text("Empty filter. Nothing done.", reply_markup=get_main_menu_keyboard())
        return MAIN_MENU
        
    merged_entries = context.user_data['merged_entries']
    commented = toggle_comment(merged_entries, filt, comment=True)
    
    content = render_clean_playlist(commented)
    await send_file_to_user(update, context, "commented.m3u", content)
    
    await update.message.reply_text(f"Commented entries matching '<b>{filt}</b>'.", reply_markup=get_main_menu_keyboard(), parse_mode=ParseMode.HTML)
    return MAIN_MENU

async def handle_uncomment_filter(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles Menu 6: Uncomment by Filter"""
    filt = update.message.text
    if not filt:
        await update.message.reply_text("Empty filter. Nothing done.", reply_markup=get_main_menu_keyboard())
        return MAIN_MENU
        
    merged_entries = context.user_data['merged_entries']
    uncommented = toggle_comment(merged_entries, filt, comment=False)
    
    content = render_clean_playlist(uncommented)
    await send_file_to_user(update, context, "uncommented.m3u", content)
    
    await update.message.reply_text(f"Un-commented entries matching '<b>{filt}</b>'.", reply_markup=get_main_menu_keyboard(), parse_mode=ParseMode.HTML)
    return MAIN_MENU

async def handle_compare_a(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles Menu 7: Upload 'A' file"""
    doc = update.message.document
    if not doc or not (doc.file_name.lower().endswith(".m3u") or doc.file_name.lower().endswith(".m3u8")):
        await update.message.reply_text("That's not an .m3u or .m3u8 file. Please upload file 'A'.")
        return AWAITING_COMPARE_A
        
    file = await context.bot.get_file(doc.file_id)
    with io.BytesIO() as b:
        await file.download_to_memory(b)
        b.seek(0)
        content_a = b.read().decode('utf-8', errors='ignore')
    
    entries_a = parse_m3u_from_content(doc.file_name, content_a)
    context.user_data['compare_a_entries'] = merge_entries([entries_a]) 
        
    await update.message.reply_text("✅ File 'A' received. Now, please upload the 'B' file (the new list).")
    return AWAITING_COMPARE_B

async def handle_compare_b(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles Menu 7: Upload 'B' file and run comparison"""
    doc = update.message.document
    if not doc or not (doc.file_name.lower().endswith(".m3u") or doc.file_name.lower().endswith(".m3u8")):
        await update.message.reply_text("That's not an .m3u or .m3u8 file. Please upload file 'B'.")
        return AWAITING_COMPARE_B
        
    file = await context.bot.get_file(doc.file_id)
    
    with io.BytesIO() as b:
        await file.download_to_memory(b)
        b.seek(0)
        content_b = b.read().decode('utf-8', errors='ignore')
        
    entries_b = parse_m3u_from_content(doc.file_name, content_b)
    entries_b_merged = merge_entries([entries_b])
    
    entries_a = context.user_data.get('compare_a_entries')
    if not entries_a:
         await update.message.reply_text("Error: File 'A' data was lost. Please start comparison over.", reply_markup=get_main_menu_keyboard())
         return MAIN_MENU
         
    # Run the comparison
    only_in_b = compare_by_name(entries_a, entries_b_merged)
    
    content = render_clean_playlist(only_in_b)
    await send_file_to_user(update, context, "compare_B_not_in_A.m3u", content)
    
    await update.message.reply_text(f"Found <b>{len(only_in_b)}</b> channels present in B but not in A.", reply_markup=get_main_menu_keyboard(), parse_mode=ParseMode.HTML)
    return MAIN_MENU

async def menu_8_export_active(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles Menu 8: Export Active-Only"""
    status_map = context.user_data.get('status_map')
    if not status_map:
        await update.message.reply_text("⚠️ You must run 'Analyze' (option 1) first to know which channels are active.", reply_markup=get_main_menu_keyboard())
        return MAIN_MENU
        
    await update.message.reply_text("Exporting Active-only numbered playlist...", reply_markup=ReplyKeyboardRemove())
    merged_entries = context.user_data['merged_entries']
    
    text = render_numbered_playlist(merged_entries, only_active=status_map)
    await send_file_to_user(update, context, "active_only_numbered.m3u", text)
    
    await update.message.reply_text("What's next?", reply_markup=get_main_menu_keyboard())
    return MAIN_MENU

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancels and ends the conversation."""
    context.user_data.clear()
    await update.message.reply_text(
        "Action cancelled. All uploaded files and data have been cleared. "
        "Send /start to begin again.",
        reply_markup=ReplyKeyboardRemove()
    )
    return ConversationHandler.END

# --- Main Bot Setup ---

def main() -> None:
    """Run the bot."""
    # Get token from environment variable
    TOKEN = "8318099986:AAFa3VRaEcJyqi-wkVMiqbF1EInqkyVzA54"
    if not TOKEN:
        print("Error: TELEGRAM_BOT_TOKEN environment variable not set.", file=sys.stderr)
        sys.exit(1)

        
    application = Application.builder().token(TOKEN).build()
    
    # Setup the conversation handler
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            SELECTING_FILES: [
                MessageHandler(filters.Document.ALL, handle_file),
                CommandHandler("done", done_uploading)
            ],
            MAIN_MENU: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_main_menu)
            ],
            AWAITING_COMMENT_FILTER: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_comment_filter)
            ],
            AWAITING_UNCOMMENT_FILTER: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_uncomment_filter)
            ],
            AWAITING_COMPARE_A: [
                 MessageHandler(filters.Document.ALL, handle_compare_a)
            ],
            AWAITING_COMPARE_B: [
                 MessageHandler(filters.Document.ALL, handle_compare_b)
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel), CommandHandler("start", start)],
        per_user=True, # Important: keeps each user's state separate
    )
    
    application.add_handler(conv_handler)
    
    print("Bot is running... Press Ctrl-C to stop.")
    application.run_polling()


if __name__ == "__main__":
    main()

# =============================================================================
#
# END: TELEGRAM BOT LOGIC
#
# ============================================================================={}
