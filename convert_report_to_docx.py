#!/usr/bin/env python3
"""
Convert PROJECT_REPORT.md to PROJECT_REPORT.docx using python-docx.
Run this script once the markdown is ready.
"""

import os
import sys
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
import re

def convert_markdown_to_docx(md_file, docx_file):
    """Convert markdown file to Word document."""
    
    # Read markdown
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Create document
    doc = Document()
    
    # Add styles
    style = doc.styles['Normal']
    style.font.name = 'Calibri'
    style.font.size = Pt(11)
    
    # Split into lines
    lines = md_content.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Skip empty lines
        if not line.strip():
            i += 1
            continue
        
        # Headers
        if line.startswith('# '):
            p = doc.add_heading(line[2:].strip(), level=1)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if i < 50 else WD_ALIGN_PARAGRAPH.LEFT
        elif line.startswith('## '):
            doc.add_heading(line[3:].strip(), level=2)
        elif line.startswith('### '):
            doc.add_heading(line[4:].strip(), level=3)
        elif line.startswith('#### '):
            doc.add_heading(line[5:].strip(), level=4)
        
        # Code blocks
        elif line.startswith('```'):
            i += 1
            code_block = []
            while i < len(lines) and not lines[i].startswith('```'):
                code_block.append(lines[i])
                i += 1
            
            # Add as paragraph with monospace
            code_text = '\n'.join(code_block).strip()
            p = doc.add_paragraph(code_text, style='List Bullet')
            p_format = p.paragraph_format
            p_format.left_indent = Inches(0.5)
            for run in p.runs:
                run.font.name = 'Courier New'
                run.font.size = Pt(9)
        
        # Tables
        elif '|' in line and line.strip().startswith('|'):
            # Simple table detection
            table_rows = []
            table_rows.append(line)
            j = i + 1
            while j < len(lines) and '|' in lines[j]:
                if '-' not in lines[j]:  # Skip header separator
                    table_rows.append(lines[j])
                j += 1
            
            if len(table_rows) > 1:
                # Parse table
                rows = [row.split('|')[1:-1] for row in table_rows]
                rows = [[cell.strip() for cell in row] for row in rows]
                
                if rows[0]:  # Check if table has content
                    cols = len(rows[0])
                    table = doc.add_table(rows=len(rows), cols=cols)
                    table.style = 'Light Grid Accent 1'
                    
                    for row_idx, row_data in enumerate(rows):
                        for col_idx, cell_data in enumerate(row_data):
                            table.rows[row_idx].cells[col_idx].text = cell_data
                
                i = j - 1
        
        # Bullet points
        elif line.startswith('- '):
            doc.add_paragraph(line[2:].strip(), style='List Bullet')
        elif line.startswith('  - '):
            doc.add_paragraph(line[4:].strip(), style='List Bullet 2')
        
        # Regular text
        elif line.strip():
            p = doc.add_paragraph(line.strip())
        
        i += 1
    
    # Save document
    doc.save(docx_file)
    print(f"✅ Successfully converted {md_file} → {docx_file}")
    print(f"   File size: {os.path.getsize(docx_file) / 1024 / 1024:.2f} MB")

if __name__ == '__main__':
    md_file = 'PROJECT_REPORT.md'
    docx_file = 'PROJECT_REPORT.docx'
    
    if not os.path.exists(md_file):
        print(f"❌ Error: {md_file} not found")
        sys.exit(1)
    
    try:
        convert_markdown_to_docx(md_file, docx_file)
        print(f"\n📄 Word document ready at: {docx_file}")
        print(f"   Pages: ~50–60 (estimated)")
        print(f"   You can now open this in MS Word or Google Docs")
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        sys.exit(1)
