#!/usr/bin/env python3
"""
Generate a professional PDF manual for Smart Traffic RL project.
Uses PyMuPDF (fitz) which is already installed.
"""
import fitz  # PyMuPDF
import os

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "Smart_Traffic_RL_Manual.pdf")

# --- Color Palette ---
DARK_BG = fitz.pdfcolor["white"]
ACCENT_BLUE = (0.18, 0.45, 0.82)
ACCENT_GREEN = (0.13, 0.72, 0.42)
ACCENT_RED = (0.88, 0.22, 0.24)
ACCENT_ORANGE = (0.93, 0.56, 0.13)
ACCENT_PURPLE = (0.49, 0.27, 0.78)
DARK_GRAY = (0.15, 0.15, 0.18)
MED_GRAY = (0.35, 0.35, 0.40)
LIGHT_GRAY = (0.92, 0.93, 0.95)
WHITE = (1, 1, 1)
TABLE_HEADER_BG = (0.12, 0.14, 0.18)
TABLE_ROW_BG = (0.96, 0.97, 0.98)
TABLE_ALT_BG = (0.90, 0.92, 0.96)
CODE_BG = (0.14, 0.16, 0.20)
CODE_TEXT = (0.85, 0.92, 0.85)
TIP_BG = (0.90, 0.96, 0.90)
TIP_BORDER = ACCENT_GREEN
WARNING_BG = (1.0, 0.95, 0.88)
WARNING_BORDER = ACCENT_ORANGE
CAUTION_BG = (1.0, 0.90, 0.90)
CAUTION_BORDER = ACCENT_RED
NOTE_BG = (0.90, 0.93, 1.0)
NOTE_BORDER = ACCENT_BLUE

PAGE_W, PAGE_H = fitz.paper_size("A4")
MARGIN_L = 55
MARGIN_R = 55
MARGIN_T = 60
MARGIN_B = 60
CONTENT_W = PAGE_W - MARGIN_L - MARGIN_R

doc = fitz.open()
current_page = None
y_cursor = MARGIN_T


def new_page():
    global current_page, y_cursor
    current_page = doc.new_page(width=PAGE_W, height=PAGE_H)
    y_cursor = MARGIN_T
    return current_page


def check_space(needed):
    global current_page, y_cursor
    if y_cursor + needed > PAGE_H - MARGIN_B:
        new_page()


def draw_text(text, fontsize=11, color=DARK_GRAY, bold=False, indent=0, spacing=4):
    global y_cursor
    fontname = "helv" if not bold else "hebo"
    check_space(fontsize + spacing + 5)
    rect = fitz.Rect(MARGIN_L + indent, y_cursor, PAGE_W - MARGIN_R, y_cursor + 200)
    # Use insert_textbox for wrapping
    rc = current_page.insert_textbox(
        rect, text, fontsize=fontsize, fontname=fontname, color=color,
        align=fitz.TEXT_ALIGN_LEFT
    )
    # Estimate lines
    avg_char_w = fontsize * 0.48
    chars_per_line = max(1, int((CONTENT_W - indent) / avg_char_w))
    num_lines = max(1, (len(text) + chars_per_line - 1) // chars_per_line)
    line_height = fontsize * 1.35
    y_cursor += num_lines * line_height + spacing


def draw_title_page():
    global y_cursor
    page = new_page()

    # Large gradient-like header block
    header_rect = fitz.Rect(0, 0, PAGE_W, PAGE_H * 0.55)
    page.draw_rect(header_rect, color=ACCENT_BLUE, fill=ACCENT_BLUE)

    # Decorative accent bar
    accent_rect = fitz.Rect(0, PAGE_H * 0.55, PAGE_W, PAGE_H * 0.565)
    page.draw_rect(accent_rect, color=ACCENT_GREEN, fill=ACCENT_GREEN)

    # Traffic light emoji / symbol
    y_emoji = 120
    page.insert_text(fitz.Point(PAGE_W / 2 - 25, y_emoji), "●", fontsize=50, color=(1, 0.2, 0.2))
    page.insert_text(fitz.Point(PAGE_W / 2 - 25, y_emoji + 55), "●", fontsize=50, color=(1, 0.85, 0.1))
    page.insert_text(fitz.Point(PAGE_W / 2 - 25, y_emoji + 110), "●", fontsize=50, color=(0.1, 0.85, 0.3))

    # Title
    title_y = PAGE_H * 0.35
    title = "Smart Traffic RL"
    tw = fitz.get_text_length(title, fontname="hebo", fontsize=38)
    page.insert_text(fitz.Point((PAGE_W - tw) / 2, title_y), title,
                     fontsize=38, fontname="hebo", color=WHITE)

    subtitle = "User Manual & Setup Guide"
    sw = fitz.get_text_length(subtitle, fontname="helv", fontsize=18)
    page.insert_text(fitz.Point((PAGE_W - sw) / 2, title_y + 40), subtitle,
                     fontsize=18, fontname="helv", color=(0.85, 0.90, 1.0))

    tag = "Multi-Agent Reinforcement Learning for Intelligent Traffic Control"
    tw2 = fitz.get_text_length(tag, fontname="helv", fontsize=12)
    page.insert_text(fitz.Point((PAGE_W - tw2) / 2, title_y + 70), tag,
                     fontsize=12, fontname="helv", color=(0.75, 0.82, 0.95))

    # Bottom info
    info_y = PAGE_H * 0.68
    infos = [
        ("Project:", "Smart Traffic Signal Control with MARL"),
        ("Environment:", "Python 3.11 + .venv Virtual Environment"),
        ("Simulator:", "SUMO 1.26.0"),
        ("RL Framework:", "Stable-Baselines3 + PettingZoo"),
        ("Dashboard:", "Streamlit + Plotly"),
    ]
    for label, value in infos:
        lw = fitz.get_text_length(label, fontname="hebo", fontsize=11)
        page.insert_text(fitz.Point(MARGIN_L + 40, info_y), label,
                         fontsize=11, fontname="hebo", color=ACCENT_BLUE)
        page.insert_text(fitz.Point(MARGIN_L + 40 + lw + 8, info_y), value,
                         fontsize=11, fontname="helv", color=MED_GRAY)
        info_y += 22

    # Footer
    footer = "Version 1.0  |  April 2026"
    fw = fitz.get_text_length(footer, fontname="helv", fontsize=10)
    page.insert_text(fitz.Point((PAGE_W - fw) / 2, PAGE_H - 50), footer,
                     fontsize=10, fontname="helv", color=MED_GRAY)


def draw_section_header(text, number=None):
    global y_cursor
    check_space(45)
    y_cursor += 12

    # Accent line
    current_page.draw_rect(
        fitz.Rect(MARGIN_L, y_cursor, MARGIN_L + CONTENT_W, y_cursor + 3),
        color=ACCENT_BLUE, fill=ACCENT_BLUE
    )
    y_cursor += 12

    prefix = f"{number}. " if number else ""
    current_page.insert_text(
        fitz.Point(MARGIN_L, y_cursor + 18),
        prefix + text, fontsize=20, fontname="hebo", color=ACCENT_BLUE
    )
    y_cursor += 32


def draw_subsection(text):
    global y_cursor
    check_space(30)
    y_cursor += 8
    current_page.insert_text(
        fitz.Point(MARGIN_L, y_cursor + 14),
        text, fontsize=14, fontname="hebo", color=DARK_GRAY
    )
    y_cursor += 24


def draw_code_block(lines, title=None):
    global y_cursor
    line_h = 15
    padding = 12
    title_h = 22 if title else 0
    block_h = len(lines) * line_h + padding * 2 + title_h + 4

    check_space(block_h + 10)

    # Background
    block_rect = fitz.Rect(MARGIN_L + 5, y_cursor, PAGE_W - MARGIN_R - 5, y_cursor + block_h)
    current_page.draw_rect(block_rect, color=CODE_BG, fill=CODE_BG)

    cy = y_cursor + padding

    if title:
        current_page.insert_text(
            fitz.Point(MARGIN_L + 18, cy + 11),
            title, fontsize=9, fontname="hebo", color=(0.55, 0.65, 0.75)
        )
        cy += title_h

    for line in lines:
        current_page.insert_text(
            fitz.Point(MARGIN_L + 18, cy + 11),
            line, fontsize=10, fontname="cour", color=CODE_TEXT
        )
        cy += line_h

    y_cursor += block_h + 8


def draw_alert_box(text, alert_type="note"):
    global y_cursor
    if alert_type == "tip":
        bg, border, label = TIP_BG, TIP_BORDER, "TIP"
    elif alert_type == "warning":
        bg, border, label = WARNING_BG, WARNING_BORDER, "WARNING"
    elif alert_type == "caution":
        bg, border, label = CAUTION_BG, CAUTION_BORDER, "CAUTION"
    else:
        bg, border, label = NOTE_BG, NOTE_BORDER, "NOTE"

    avg_char_w = 10 * 0.48
    chars_per_line = max(1, int((CONTENT_W - 50) / avg_char_w))
    num_lines = max(1, (len(text) + chars_per_line - 1) // chars_per_line)
    box_h = num_lines * 15 + 35

    check_space(box_h + 10)

    box_rect = fitz.Rect(MARGIN_L + 5, y_cursor, PAGE_W - MARGIN_R - 5, y_cursor + box_h)
    current_page.draw_rect(box_rect, color=border, fill=bg)

    # Left accent bar
    current_page.draw_rect(
        fitz.Rect(MARGIN_L + 5, y_cursor, MARGIN_L + 10, y_cursor + box_h),
        color=border, fill=border
    )

    # Label
    current_page.insert_text(
        fitz.Point(MARGIN_L + 20, y_cursor + 16),
        label, fontsize=9, fontname="hebo", color=border
    )

    # Text
    text_rect = fitz.Rect(MARGIN_L + 20, y_cursor + 22, PAGE_W - MARGIN_R - 15, y_cursor + box_h - 5)
    current_page.insert_textbox(text_rect, text, fontsize=10, fontname="helv", color=DARK_GRAY)

    y_cursor += box_h + 8


def draw_table(headers, rows, col_widths=None):
    global y_cursor
    row_h = 24
    header_h = 28
    total_h = header_h + len(rows) * row_h + 4

    check_space(total_h + 10)

    if col_widths is None:
        col_w = CONTENT_W / len(headers)
        col_widths = [col_w] * len(headers)

    x_start = MARGIN_L + 5
    table_w = sum(col_widths)

    # Header
    hdr_rect = fitz.Rect(x_start, y_cursor, x_start + table_w, y_cursor + header_h)
    current_page.draw_rect(hdr_rect, color=TABLE_HEADER_BG, fill=TABLE_HEADER_BG)

    cx = x_start
    for i, h in enumerate(headers):
        current_page.insert_text(
            fitz.Point(cx + 8, y_cursor + 18),
            h, fontsize=10, fontname="hebo", color=WHITE
        )
        cx += col_widths[i]

    y_cursor += header_h

    # Data rows
    for ri, row in enumerate(rows):
        row_bg = TABLE_ROW_BG if ri % 2 == 0 else TABLE_ALT_BG
        row_rect = fitz.Rect(x_start, y_cursor, x_start + table_w, y_cursor + row_h)
        current_page.draw_rect(row_rect, color=row_bg, fill=row_bg)

        cx = x_start
        for ci, cell in enumerate(row):
            current_page.insert_text(
                fitz.Point(cx + 8, y_cursor + 16),
                str(cell), fontsize=9.5, fontname="helv", color=DARK_GRAY
            )
            cx += col_widths[ci]
        y_cursor += row_h

    y_cursor += 10


def draw_numbered_step(number, title, description=""):
    global y_cursor
    check_space(40)

    # Circle with number
    cx_circle = MARGIN_L + 14
    cy_circle = y_cursor + 12
    current_page.draw_circle(fitz.Point(cx_circle, cy_circle), 12,
                             color=ACCENT_BLUE, fill=ACCENT_BLUE)
    nw = fitz.get_text_length(str(number), fontname="hebo", fontsize=12)
    current_page.insert_text(fitz.Point(cx_circle - nw / 2, cy_circle + 5),
                             str(number), fontsize=12, fontname="hebo", color=WHITE)

    # Title
    current_page.insert_text(
        fitz.Point(MARGIN_L + 34, y_cursor + 16),
        title, fontsize=13, fontname="hebo", color=DARK_GRAY
    )
    y_cursor += 28

    if description:
        draw_text(description, fontsize=10, indent=34, color=MED_GRAY)


def add_page_numbers():
    """Add page numbers to all pages except the title page."""
    for i in range(1, len(doc)):
        page = doc[i]
        text = f"Page {i}"
        tw = fitz.get_text_length(text, fontname="helv", fontsize=9)
        page.insert_text(
            fitz.Point((PAGE_W - tw) / 2, PAGE_H - 30),
            text, fontsize=9, fontname="helv", color=MED_GRAY
        )
        # Header line
        page.draw_line(
            fitz.Point(MARGIN_L, 42), fitz.Point(PAGE_W - MARGIN_R, 42),
            color=(0.85, 0.87, 0.90), width=0.5
        )
        header_text = "Smart Traffic RL  |  User Manual"
        page.insert_text(fitz.Point(MARGIN_L, 37), header_text,
                         fontsize=8, fontname="helv", color=MED_GRAY)


# ============================================================
# BUILD THE DOCUMENT
# ============================================================

# --- TITLE PAGE ---
draw_title_page()

# --- TABLE OF CONTENTS ---
new_page()
draw_section_header("Table of Contents")
toc_items = [
    ("1.", "Prerequisites & System Requirements"),
    ("2.", "Environment Setup (Which Environment to Choose)"),
    ("3.", "Activating the Virtual Environment"),
    ("4.", "Running the Project"),
    ("5.", "Menu Options Explained"),
    ("6.", "Recommended First-Time Run Order"),
    ("7.", "Direct Command-Line Usage"),
    ("8.", "Troubleshooting"),
]
for num, title in toc_items:
    draw_text(f"  {num}   {title}", fontsize=12, color=DARK_GRAY, spacing=8)

# --- SECTION 1: PREREQUISITES ---
new_page()
draw_section_header("Prerequisites & System Requirements", "1")

draw_text("Before running the project, ensure the following software is installed on your system:",
          fontsize=11, color=MED_GRAY)
y_cursor += 5

draw_table(
    ["Requirement", "Required Version", "Purpose"],
    [
        ["Python", "3.10 or 3.11 (NOT 3.14)", "Core runtime"],
        ["SUMO Simulator", "1.19+ (recommended 1.26)", "Traffic simulation engine"],
        ["pip", "Latest", "Package manager"],
        ["PowerShell", "5.1+", "Terminal (Windows)"],
    ],
    [140, 170, CONTENT_W - 310 - 10]
)

draw_alert_box(
    "Your system has Python 3.14 as default, but PyTorch does NOT support Python 3.14. "
    "The project includes a .venv virtual environment with Python 3.11.8 which you MUST use.",
    "caution"
)

draw_subsection("Installed Packages (in .venv)")
draw_text("All required packages are pre-installed in the .venv virtual environment:", fontsize=10, color=MED_GRAY)

draw_table(
    ["Package", "Version", "Purpose"],
    [
        ["torch", "2.11.0", "Deep learning framework"],
        ["stable-baselines3", "2.8.0", "RL algorithms (PPO)"],
        ["gymnasium", "1.2.3", "RL environment interface"],
        ["pettingzoo", "1.24.1", "Multi-agent environments"],
        ["supersuit", "3.9.0", "Multi-agent wrappers"],
        ["streamlit", "1.56.0", "Web dashboard"],
        ["plotly", "6.7.0", "Interactive charts"],
        ["numpy", "2.4.4", "Numerical computing"],
        ["pandas", "3.0.2", "Data processing"],
        ["traci", "1.26.0", "SUMO Python interface"],
        ["matplotlib", "3.10.8", "Static plotting"],
    ],
    [140, 100, CONTENT_W - 240 - 10]
)

# --- SECTION 2: ENVIRONMENT SETUP ---
new_page()
draw_section_header("Environment Setup", "2")
draw_subsection("Which Environment to Choose?")

draw_text(
    "Your project directory contains multiple virtual environment folders. "
    "Here is what each one is and which one you should use:",
    fontsize=11, color=MED_GRAY
)
y_cursor += 5

draw_table(
    ["Folder", "Python Version", "Status", "Use This?"],
    [
        [".venv", "3.11.8", "All packages installed", "YES"],
        ["venv", "3.11.8", "Incomplete setup", "NO"],
        ["venv_old", "Unknown", "Outdated / broken", "NO"],
        ["System Python", "3.14.0", "Incompatible with PyTorch", "NO"],
    ],
    [100, 110, 160, 100]
)

draw_alert_box(
    "ALWAYS use the .venv folder (Python 3.11.8). This is the only environment "
    "with all dependencies correctly installed. The system Python 3.14 will cause "
    "import errors with PyTorch and other packages.",
    "caution"
)

draw_subsection("Why Python 3.11?")
draw_text(
    "PyTorch (the deep learning backend) and several scientific computing libraries "
    "require Python 3.10-3.12. Python 3.14 is too new and not yet supported by these "
    "libraries. Python 3.11 offers the best compatibility and performance.",
    fontsize=10.5, color=MED_GRAY
)

draw_subsection("SUMO Simulator Path")
draw_text("SUMO is installed at the following location. The main.py script automatically configures this:",
          fontsize=10.5, color=MED_GRAY)
draw_code_block([
    r'SUMO_HOME = C:\Program Files (x86)\Eclipse\Sumo',
    r'SUMO_BIN  = C:\Program Files (x86)\Eclipse\Sumo\bin',
])

# --- SECTION 3: ACTIVATING THE ENVIRONMENT ---
new_page()
draw_section_header("Activating the Virtual Environment", "3")

draw_text(
    "Follow these steps every time you want to run the project. "
    "You must activate the virtual environment before running any Python command.",
    fontsize=11, color=MED_GRAY
)
y_cursor += 5

draw_numbered_step(1, "Open PowerShell / VS Code Terminal",
                   "Open a terminal window. In VS Code, press Ctrl+` (backtick) to open the integrated terminal.")

draw_numbered_step(2, "Navigate to the Project Directory")
draw_code_block([
    r'cd "c:\Users\vipin\Desktop\NITIN_SAROJ\projects\ml_project\smart-traffic-rl"'
], title="PowerShell")

draw_numbered_step(3, "Activate the .venv Environment")
draw_code_block([
    r'.\.venv\Scripts\Activate.ps1'
], title="PowerShell")

draw_text(
    "After activation, your terminal prompt will change to show (.venv) at the beginning:",
    fontsize=10, color=MED_GRAY, indent=34
)
draw_code_block([
    r'(.venv) PS C:\Users\vipin\Desktop\...\smart-traffic-rl>'
])

draw_numbered_step(4, "Verify the Python Version")
draw_code_block([
    'python --version',
    '# Expected output: Python 3.11.8'
], title="Verification")

draw_alert_box(
    "If you get an 'execution policy' error when activating, run this command once:\n"
    "Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned",
    "tip"
)

# --- SECTION 4: RUNNING THE PROJECT ---
new_page()
draw_section_header("Running the Project", "4")

draw_subsection("Option A: Interactive Menu (Recommended)")
draw_text(
    "The easiest way to run the project. This shows a menu with all available options:",
    fontsize=11, color=MED_GRAY
)

draw_code_block([
    'python main.py'
], title="Command")

draw_text("The menu will display:", fontsize=10, color=MED_GRAY)
draw_code_block([
    '+-------------------------------------------------------------+',
    '|                       MAIN MENU                             |',
    '+-------------------------------------------------------------+',
    '|  1. Train Single Agent (PPO with Curriculum Learning)       |',
    '|  2. Train Multi-Agent (MAPPO for 9 Intersections)           |',
    '|  3. Run Dashboard (Real-time Visualization)                 |',
    '|  4. Run Evaluation Only                                     |',
    '|  5. Run Adversarial Tests                                   |',
    '|  6. Generate SUMO Network                                   |',
    '|  7. Run All Tests                                           |',
    '|  8. Exit                                                    |',
    '+-------------------------------------------------------------+',
], title="Interactive Menu")

draw_text("Type a number (1-8) and press Enter to select an option.", fontsize=10.5, color=MED_GRAY)

draw_subsection("Option B: Command-Line Flags (Quick Access)")
draw_text("You can bypass the menu by passing a --mode flag:", fontsize=11, color=MED_GRAY)

draw_code_block([
    'python main.py --mode train       # Train single agent',
    'python main.py --mode multi       # Train multi-agent',
    'python main.py --mode dashboard   # Launch dashboard',
    'python main.py --mode eval        # Run evaluation',
    'python main.py --mode test        # Run all tests',
], title="Direct Commands")

# --- SECTION 5: MENU OPTIONS EXPLAINED ---
new_page()
draw_section_header("Menu Options Explained", "5")

# Option 1
draw_subsection("Option 1: Train Single Agent")
draw_text(
    "Trains a PPO (Proximal Policy Optimization) agent on the center intersection (n11) "
    "using curriculum learning. The training progresses through 3 stages of increasing difficulty:",
    fontsize=10.5, color=MED_GRAY
)

draw_table(
    ["Stage", "Timesteps", "Max Vehicles", "Difficulty"],
    [
        ["Stage 1", "50,000", "100", "Easy"],
        ["Stage 2", "100,000", "300", "Medium"],
        ["Stage 3", "150,000", "500", "Hard"],
    ],
    [80, 120, 120, 150]
)

draw_text("Estimated time: 30-60 minutes. Model saved to: agents/ppo_final.zip",
          fontsize=10, color=MED_GRAY, bold=True)

# Option 2
y_cursor += 5
draw_subsection("Option 2: Train Multi-Agent (MAPPO)")
draw_text(
    "Trains 9 independent MAPPO agents (one per intersection) on the full 3x3 grid. "
    "Uses the single-agent model as a starting point if available (transfer learning). "
    "Each agent manages its own traffic signal while cooperating with neighbors.",
    fontsize=10.5, color=MED_GRAY
)
draw_text("Estimated time: 1-3 hours. Model saved to: agents/mappo_final.zip",
          fontsize=10, color=MED_GRAY, bold=True)

draw_alert_box(
    "Train the single agent FIRST (Option 1), then the multi-agent (Option 2). "
    "The multi-agent system uses transfer learning from the single agent's weights.",
    "warning"
)

# Option 3
draw_subsection("Option 3: Run Dashboard")
draw_text(
    "Launches a Streamlit web dashboard for real-time monitoring. The dashboard shows "
    "traffic metrics, signal phases, queue lengths, CO2 emissions, and explainable AI "
    "decision rationale. Opens at http://localhost:8501 in your browser.",
    fontsize=10.5, color=MED_GRAY
)
draw_text("Press Ctrl+C in the terminal to stop the dashboard.", fontsize=10, color=ACCENT_RED, bold=True)

# Options 4-7
check_space(180)
draw_subsection("Option 4: Run Evaluation")
draw_text("Evaluates the trained models against a fixed-timing baseline to measure improvement.",
          fontsize=10.5, color=MED_GRAY)

draw_subsection("Option 5: Run Adversarial Tests")
draw_text("Tests system robustness against: lane blockages (accidents), sensor failures, "
          "and multiple emergency vehicles simultaneously.",
          fontsize=10.5, color=MED_GRAY)

draw_subsection("Option 6: Generate SUMO Network")
draw_text("Creates the 3x3 intersection grid network files. Run this FIRST if the simulation "
          "files (grid_network.net.xml, routes.rou.xml) don't exist yet.",
          fontsize=10.5, color=MED_GRAY)

draw_subsection("Option 7: Run All Tests")
draw_text("Executes the complete test suite: SUMO connectivity, environment validation, "
          "and adversarial robustness tests.",
          fontsize=10.5, color=MED_GRAY)

# --- SECTION 6: RECOMMENDED RUN ORDER ---
new_page()
draw_section_header("Recommended First-Time Run Order", "6")

draw_text(
    "When running the project for the first time, follow this exact order. "
    "Each step depends on the previous one:",
    fontsize=11, color=MED_GRAY
)
y_cursor += 10

steps = [
    ("Activate .venv", ".\\. venv\\Scripts\\Activate.ps1", "Sets up the correct Python 3.11 environment"),
    ("Generate SUMO Network", "python main.py  ->  Option 6", "Creates the 3x3 grid simulation files"),
    ("Train Single Agent", "python main.py  ->  Option 1", "Trains PPO on center intersection (~30-60 min)"),
    ("Train Multi-Agent", "python main.py  ->  Option 2", "Trains 9 MAPPO agents on full grid (~1-3 hrs)"),
    ("Launch Dashboard", "python main.py  ->  Option 3", "View results at http://localhost:8501"),
    ("Run Tests", "python main.py  ->  Option 5 or 7", "Verify adversarial robustness"),
]

for i, (title, cmd, desc) in enumerate(steps, 1):
    draw_numbered_step(i, title, desc)
    if "venv" not in cmd:
        draw_code_block([cmd.replace("->", "→")])

draw_alert_box(
    "Steps 3 and 4 (training) are computationally intensive. A GPU will significantly "
    "speed up training. If you only have a CPU, expect longer training times but the "
    "system will still work correctly.",
    "note"
)

# --- SECTION 7: DIRECT COMMAND LINE ---
new_page()
draw_section_header("Direct Command-Line Usage", "7")

draw_subsection("Quick Reference Commands")
draw_text("Copy-paste these commands after activating .venv:", fontsize=10.5, color=MED_GRAY)

draw_code_block([
    '# Interactive menu',
    'python main.py',
    '',
    '# Train single agent',
    'python main.py --mode train',
    '',
    '# Train multi-agent',
    'python main.py --mode multi',
    '',
    '# Launch dashboard',
    'python main.py --mode dashboard',
    '',
    '# Alternative dashboard launch',
    'python run_dashboard.py',
    'python -m streamlit run dashboard/app.py',
    '',
    '# Run tests',
    'python main.py --mode test',
], title="All Available Commands")

draw_subsection("Project File Structure")
draw_code_block([
    'smart-traffic-rl/',
    '  main.py              Main entry point (run this)',
    '  config.py            Central configuration',
    '  requirements.txt     Package dependencies',
    '  run_dashboard.py     Dashboard launcher shortcut',
    '  .venv/               Virtual environment (Python 3.11)',
    '  agents/              Trained model files (.zip)',
    '  env/                 Gymnasium & PettingZoo environments',
    '  training/            Training scripts & results',
    '  simulation/          SUMO network & route files',
    '  dashboard/           Streamlit visualization app',
    '  tests/               Test suites',
], title="Project Structure")

# --- SECTION 8: TROUBLESHOOTING ---
new_page()
draw_section_header("Troubleshooting", "8")

draw_text("Common issues and their solutions:", fontsize=11, color=MED_GRAY)
y_cursor += 5

problems = [
    ["ModuleNotFoundError", "Forgot to activate .venv", ".\\. venv\\Scripts\\Activate.ps1"],
    ["Python 3.14 showing", "Wrong Python active", "Activate .venv first"],
    ["Execution policy error", "PowerShell restriction", "Set-ExecutionPolicy -Scope CurrentUser RemoteSigned"],
    ["SUMO not found", "Missing simulator", "Install from sumo.dlr.de"],
    ["Dashboard won't open", "Port conflict", "Open http://localhost:8501 manually"],
    ["Training is very slow", "No GPU / normal behavior", "Use GPU if available; CPU works too"],
    ["Permission denied", "Admin rights needed", "Run terminal as Administrator"],
    ["traci error", "SUMO_HOME not set", "main.py sets this automatically"],
]

draw_table(
    ["Problem", "Cause", "Solution"],
    problems,
    [150, 140, CONTENT_W - 290 - 10]
)

y_cursor += 10
draw_alert_box(
    "The most common issue is forgetting to activate .venv before running commands. "
    "Always check that (.venv) appears in your terminal prompt before running python.",
    "tip"
)

y_cursor += 10
draw_subsection("Getting Help")
draw_text(
    "If you encounter issues not covered here, check the README.md in the project root "
    "for additional documentation, or review the error messages carefully - they often "
    "indicate exactly what is missing or misconfigured.",
    fontsize=10.5, color=MED_GRAY
)

# --- Add page numbers ---
add_page_numbers()

# --- Save ---
doc.save(OUTPUT_PATH)
doc.close()
print(f"\nPDF Manual generated successfully!")
print(f"Saved to: {OUTPUT_PATH}")
print(f"Total pages: {len(doc) if doc.is_closed else 'saved'}")
