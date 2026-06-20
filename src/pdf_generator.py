import io
import os
import tempfile
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams
matplotlib.use("Agg")
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.platypus import KeepTogether

# ── PREMIUM MODERN COLORS ──
C_BG = colors.HexColor("#F8FAFC")
C_PRIMARY = colors.HexColor("#0F172A")       # Slate 900
C_ACCENT = colors.HexColor("#4F46E5")        # Indigo 600
C_ACCENT_LIGHT = colors.HexColor("#EEF2FF")  # Indigo 50
C_SUCCESS = colors.HexColor("#10B981")       # Emerald 500
C_WARNING = colors.HexColor("#F59E0B")       # Amber 500
C_DANGER = colors.HexColor("#EF4444")        # Red 500
C_CARD_BG = colors.white
C_BORDER = colors.HexColor("#E2E8F0")        # Slate 200
C_TEXT = colors.HexColor("#334155")          # Slate 700
C_TEXT_MUTED = colors.HexColor("#64748B")    # Slate 500

def set_plot_style():
    """Applies a clean, modern style to matplotlib plots."""
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'sans-serif']
    rcParams['axes.spines.top'] = False
    rcParams['axes.spines.right'] = False
    rcParams['axes.spines.left'] = False
    rcParams['axes.spines.bottom'] = False
    rcParams['axes.prop_cycle'] = matplotlib.cycler(color=['#4F46E5', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'])
    rcParams['axes.edgecolor'] = '#CBD5E1'
    rcParams['text.color'] = '#334155'

def _create_cost_pie_chart(plan):
    set_plot_style()
    cost_data = plan.get('estimated_cost', {})
    labels = ['Manpower', 'Barricades', 'Equipment']
    sizes = [
        cost_data.get('manpower_cost_inr', 0),
        cost_data.get('barricade_cost_inr', 0),
        cost_data.get('equipment_cost_inr', 0)
    ]
    if sum(sizes) == 0:
        return None
        
    fig, ax = plt.subplots(figsize=(5, 3.5))
    
    # Modern Donut Chart
    wedges, texts, autotexts = ax.pie(
        sizes, 
        labels=labels, 
        autopct='%1.1f%%', 
        startangle=90, 
        colors=['#4F46E5', '#F59E0B', '#10B981'],
        wedgeprops={'width': 0.4, 'edgecolor': 'w', 'linewidth': 3}, 
        textprops={'fontsize': 10, 'weight': 'bold', 'color': '#334155'}
    )
    
    # Improve label styling
    for text in texts:
        text.set_color('#64748B')
        text.set_fontsize(9)
        text.set_weight('normal')
        
    ax.axis('equal')
    plt.title("Financial Breakdown", fontsize=12, fontweight='bold', color='#0F172A', pad=15)
    
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp.name, bbox_inches='tight', dpi=200, transparent=True)
    plt.close(fig)
    return tmp.name

def _create_map_plot(lat, lon):
    set_plot_style()
    if not lat or not lon:
        return None
    fig, ax = plt.subplots(figsize=(5, 3.5))
    
    # Draw a stylized grid representation
    ax.plot(lon, lat, marker='o', markersize=25, color='#EF4444', alpha=0.15)
    ax.plot(lon, lat, marker='o', markersize=10, color='#EF4444')
    ax.plot(lon, lat, marker='o', markersize=4, color='white')
    
    ax.set_xlim(lon - 0.015, lon + 0.015)
    ax.set_ylim(lat - 0.015, lat + 0.015)
    
    # Clean grid lines
    ax.grid(True, linestyle='-', alpha=0.08, color='#0F172A', linewidth=1.5)
    
    # Remove axis ticks entirely for a cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax.set_title("Incident Location Coordinates", fontsize=12, fontweight='bold', color='#0F172A', pad=15)
    ax.text(0.5, -0.08, f"LATITUDE: {lat:.5f}  |  LONGITUDE: {lon:.5f}", 
            transform=ax.transAxes, ha='center', fontsize=8, color='#64748B', fontweight='bold')
    
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp.name, bbox_inches='tight', dpi=200, transparent=True)
    plt.close(fig)
    return tmp.name

def add_page_decorations(canvas, doc):
    """Draws a premium header banner and footer on every page."""
    canvas.saveState()
    
    # Top banner line
    canvas.setStrokeColor(C_ACCENT)
    canvas.setLineWidth(6)
    canvas.line(0, letter[1], letter[0], letter[1])
    
    # Header text
    canvas.setFont('Helvetica-Bold', 8)
    canvas.setFillColor(C_TEXT_MUTED)
    canvas.drawString(40, letter[1] - 25, "BENGALURU TRAFFIC COMMAND CENTER")
    
    # Page Number
    page_num = canvas.getPageNumber()
    canvas.drawString(letter[0] - 80, letter[1] - 25, f"PAGE {page_num:02d}")
    
    canvas.restoreState()

def create_card_table(data_rows, col_widths):
    """Helper to create tables that look like modern dashboard cards."""
    t = Table(data_rows, colWidths=col_widths)
    t.setStyle(TableStyle([
        # Header row styling (Indigo accent)
        ('BACKGROUND', (0,0), (-1,0), C_ACCENT_LIGHT),
        ('TEXTCOLOR', (0,0), (-1,0), C_ACCENT),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 10),
        ('BOTTOMPADDING', (0,0), (-1,0), 10),
        ('TOPPADDING', (0,0), (-1,0), 10),
        
        # Body styling
        ('FONTNAME', (0,1), (0,-1), 'Helvetica-Bold'), # Left column bold
        ('FONTSIZE', (0,1), (-1,-1), 9),
        ('TEXTCOLOR', (0,1), (0,-1), C_PRIMARY),
        ('TEXTCOLOR', (1,1), (-1,-1), C_TEXT),
        ('BOTTOMPADDING', (0,1), (-1,-1), 8),
        ('TOPPADDING', (0,1), (-1,-1), 8),
        
        # Subtle borders
        ('LINEBELOW', (0,0), (-1,-2), 1, C_BORDER),
        ('BOX', (0,0), (-1,-1), 1, C_BORDER),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    return t

def generate_action_plan_pdf(event_profile, predictions, plan):
    """Generates an ultra-premium, dashboard-styled PDF report."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=letter,
        rightMargin=40, leftMargin=40,
        topMargin=50, bottomMargin=50
    )
    
    styles = getSampleStyleSheet()
    
    # Modern Paragraph Styles
    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Heading1"], 
        fontSize=24, fontName="Helvetica-Bold", textColor=C_PRIMARY, 
        alignment=0, spaceAfter=8, leading=28
    )
    subtitle_style = ParagraphStyle(
        "ReportSubTitle", parent=styles["Normal"], 
        fontSize=11, fontName="Helvetica-Oblique", textColor=C_TEXT_MUTED, 
        spaceAfter=35, leading=16
    )
    h2_style = ParagraphStyle(
        "SectionHeader", parent=styles["Heading2"], 
        fontSize=14, fontName="Helvetica-Bold", textColor=C_PRIMARY, 
        spaceBefore=25, spaceAfter=12, leading=18,
    )
    normal_style = ParagraphStyle(
        "ModernNormal", parent=styles["Normal"],
        fontSize=9, textColor=C_TEXT, leading=14
    )
    
    elements = []
    temp_files = []
    
    # ── 1. HEADER ──
    elements.append(Paragraph("TRAFFIC INCIDENT INTELLIGENCE REPORT", title_style))
    elements.append(Paragraph(
        f"Automated Event Prediction & Strategic Action Plan<br/>"
        f"<b>Generated:</b> {datetime.now().strftime('%d %b %Y, %H:%M:%S IST')} &nbsp;&nbsp;|&nbsp;&nbsp; <b>Status:</b> <font color='#EF4444'>CONFIDENTIAL</font>", 
        subtitle_style
    ))
    
    # ── 2. EVENT DETAILS & MAP (SIDE-BY-SIDE) ──
    elements.append(Paragraph("1. INCIDENT PROFILE", h2_style))
    
    event_data = [
        ["Attribute", "Analyzed Value"],
        ["Incident Type", str(event_profile.get('event_cause', 'Unknown')).replace('_', ' ').upper()],
        ["Affected Corridor", str(event_profile.get('corridor', 'Unknown')).upper()],
        ["Key Junction", str(event_profile.get('junction', 'Unknown')).upper()],
        ["Time Context", "RUSH HOUR" if event_profile.get('is_rush_hour') else "OFF-PEAK"],
        ["Classification", "PLANNED EVENT" if event_profile.get('is_planned') else "UNPLANNED INCIDENT"]
    ]
    t_event = create_card_table(event_data, [130, 190])
    
    lat = event_profile.get('latitude')
    lon = event_profile.get('longitude')
    map_img_path = _create_map_plot(lat, lon)
    if map_img_path:
        temp_files.append(map_img_path)
        map_img = Image(map_img_path, width=200, height=140)
        # Wrap map in a subtle card
        t_map = Table([[map_img]])
        t_map.setStyle(TableStyle([
            ('BOX', (0,0), (-1,-1), 1, C_BORDER),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2),
            ('TOPPADDING', (0,0), (-1,-1), 2),
        ]))
        
        layout_top = Table([[t_event, t_map]], colWidths=[330, 210])
        layout_top.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'TOP')]))
        elements.append(layout_top)
    else:
        elements.append(t_event)
        
    elements.append(Spacer(1, 10))
    
    # ── 3. PREDICTIONS ──
    elements.append(Paragraph("2. AI IMPACT PREDICTIONS", h2_style))
    
    sev_proba = predictions.get('severity_proba', 0)
    sev_color = "#EF4444" if sev_proba > 0.6 else "#F59E0B" if sev_proba > 0.4 else "#10B981"
    sev_label = f"<font color='{sev_color}'><b>{predictions.get('severity', 'Unknown').upper()}</b></font> &nbsp; ({sev_proba:.1%} Confidence)"
    
    dur_val = predictions.get('duration_hours', {}).get('p50', 0.0)
    clos_needed = predictions.get('closure_needed')
    clos_val = "<font color='#EF4444'><b>REQUIRED</b></font>" if clos_needed else "<font color='#10B981'><b>NOT REQUIRED</b></font>"
    
    alert = plan.get('alert_level', {}) if plan else {}
    alert_col = alert.get('color', 'N/A')
    alert_hex = {'RED': '#EF4444', 'ORANGE': '#F97316', 'YELLOW': '#F59E0B', 'BLUE': '#3B82F6', 'GREEN': '#10B981'}.get(alert_col, '#334155')
    alert_text = f"<font color='{alert_hex}'><b>LEVEL {alert.get('level', 'N/A')} &mdash; {alert.get('description', 'N/A').upper()}</b></font>"
    
    pred_data = [
        ["Key Impact Metric", "AI Assessment"],
        ["Severity Level", Paragraph(sev_label, normal_style)],
        ["Estimated Duration", f"{dur_val:.1f} Hours"],
        ["Road Closure", Paragraph(clos_val, normal_style)],
        ["Command Alert Status", Paragraph(alert_text, normal_style)]
    ]
    t_pred = create_card_table(pred_data, [130, 400])
    elements.append(t_pred)
    elements.append(Spacer(1, 10))
    
    # ── 4. ACTION PLAN ──
    elements.append(Paragraph("3. STRATEGIC DEPLOYMENT PLAN", h2_style))
    
    if plan:
        diversion = plan.get('diversion', {})
        if diversion:
            div_data = [
                ["Routing Directives", "Path"],
                ["Primary Route", Paragraph(diversion.get('primary', 'N/A'), normal_style)],
                ["Secondary Route", Paragraph(diversion.get('secondary', 'N/A'), normal_style)],
                ["Strictly Avoid", Paragraph(f"<font color='#EF4444'><b>{diversion.get('avoid', 'N/A')}</b></font>", normal_style)]
            ]
            t_div = create_card_table(div_data, [130, 400])
            elements.append(t_div)
            elements.append(Spacer(1, 15))
            
        # Resource & Cost Section Side-by-Side
        manpower = plan.get('manpower', {})
        equipment = plan.get('equipment', [])
        
        mp_lines = [f"• {role.replace('_', ' ').title()}: <b>{count}</b>" for role, count in manpower.items() if role != "formula"]
        eq_lines = [f"• {item}" for item in equipment]
        
        res_data = [
            ["Deployment Type", "Details"],
            ["Personnel", Paragraph("<br/>".join(mp_lines), normal_style)],
            ["Equipment", Paragraph("<br/>".join(eq_lines) if eq_lines else "None Required", normal_style)]
        ]
        t_res = create_card_table(res_data, [130, 180])
        
        # Financial Table
        cost_data = plan.get('estimated_cost', {})
        c_table_data = [
            ["Cost Category", "Amount (INR)"],
            ["Manpower", f"₹ {cost_data.get('manpower_cost_inr', 0):,}"],
            ["Barricading", f"₹ {cost_data.get('barricade_cost_inr', 0):,}"],
            ["Equipment", f"₹ {cost_data.get('equipment_cost_inr', 0):,}"]
        ]
        t_cost = create_card_table(c_table_data, [120, 100])
        
        # Add Total Row with solid primary background
        c_table_data.append(["Total Estimate", f"₹ {cost_data.get('total_estimated_cost_inr', 0):,}"])
        t_cost = Table(c_table_data, colWidths=[120, 100])
        
        # Duplicate the card style but modify the last row
        t_cost_style = [
            ('BACKGROUND', (0,0), (-1,0), C_ACCENT_LIGHT),
            ('TEXTCOLOR', (0,0), (-1,0), C_ACCENT),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 10),
            ('BOTTOMPADDING', (0,0), (-1,0), 10),
            ('TOPPADDING', (0,0), (-1,0), 10),
            ('FONTNAME', (0,1), (0,-2), 'Helvetica-Bold'),
            ('FONTSIZE', (0,1), (-1,-1), 9),
            ('TEXTCOLOR', (0,1), (0,-2), C_PRIMARY),
            ('TEXTCOLOR', (1,1), (-1,-2), C_TEXT),
            ('BOTTOMPADDING', (0,1), (-1,-1), 8),
            ('TOPPADDING', (0,1), (-1,-1), 8),
            ('LINEBELOW', (0,0), (-1,-2), 1, C_BORDER),
            ('BOX', (0,0), (-1,-1), 1, C_BORDER),
            
            # Total Row Styling
            ('BACKGROUND', (0,-1), (-1,-1), C_PRIMARY),
            ('TEXTCOLOR', (0,-1), (-1,-1), colors.white),
            ('FONTNAME', (0,-1), (-1,-1), 'Helvetica-Bold'),
        ]
        t_cost.setStyle(TableStyle(t_cost_style))
        
        pie_path = _create_cost_pie_chart(plan)
        if pie_path:
            temp_files.append(pie_path)
            pie_img = Image(pie_path, width=200, height=140)
            
            # Stack Cost table and Pie chart vertically in the right column
            cost_stack = Table([[pie_img], [Spacer(1, 5)], [t_cost]], colWidths=[230])
            cost_stack.setStyle(TableStyle([
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('BOTTOMPADDING', (0,0), (-1,-1), 0),
                ('TOPPADDING', (0,0), (-1,-1), 0),
            ]))
            
            layout_bottom = Table([[t_res, cost_stack]], colWidths=[320, 240])
            layout_bottom.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'TOP')]))
            elements.append(KeepTogether(layout_bottom))
        else:
            elements.append(t_res)
            elements.append(Spacer(1, 10))
            elements.append(t_cost)
            
    else:
        elements.append(Paragraph("No action plan available for this event.", normal_style))

    # Build the PDF with page numbers
    doc.build(elements, onFirstPage=add_page_decorations, onLaterPages=add_page_decorations)
    
    # Cleanup temp files
    for f in temp_files:
        try:
            os.remove(f)
        except Exception:
            pass
            
    buffer.seek(0)
    return buffer
