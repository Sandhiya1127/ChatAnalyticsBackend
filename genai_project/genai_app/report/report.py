import os
import base64
import json
import textwrap
import pandas as pd
from django.http import FileResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from fpdf import FPDF
import tempfile
import uuid
import re
def clean_unicode_text(text):
    """Clean text by replacing problematic Unicode characters"""
    if not isinstance(text, str):
        text = str(text)
    
    # Handle None or empty values
    if not text or text.strip() == '':
        return ''
    
    # Replace common problematic Unicode characters
    replacements = {
        '₹': 'INR ',     # Indian Rupee
        '€': 'EUR ',     # Euro
        '£': 'GBP ',     # Pound
        '¥': 'JPY ',     # Yen
        '$': 'USD ',     # Dollar (if causing issues)
        '°': ' deg ',    # Degree symbol
        '–': '-',        # En dash
        '—': '-',        # Em dash
        ''': "'",        # Left single quotation
        ''': "'",        # Right single quotation
        '"': '"',        # Left double quotation
        '"': '"',        # Right double quotation
        '…': '...',      # Ellipsis
        '™': '(TM)',     # Trademark
        '®': '(R)',      # Registered trademark
        '©': '(C)',      # Copyright
        '±': '+/-',      # Plus-minus
        '×': 'x',        # Multiplication sign
        '÷': '/',        # Division sign
        '≤': '<=',       # Less than or equal
        '≥': '>=',       # Greater than or equal
        '≠': '!=',       # Not equal
        '√': 'sqrt',     # Square root
        '∞': 'infinity', # Infinity
        '∑': 'sum',      # Summation
        '∆': 'delta',    # Delta
        'α': 'alpha',    # Greek letters commonly used in data
        'β': 'beta',
        'γ': 'gamma',
        'δ': 'delta',
        'π': 'pi',
        'σ': 'sigma',
        'μ': 'mu',
        'λ': 'lambda',
    }
    
    # Apply replacements
    for unicode_char, replacement in replacements.items():
        text = text.replace(unicode_char, replacement)
    
    # Remove any remaining problematic Unicode characters
    # Keep only printable ASCII and common extended ASCII
    text = re.sub(r'[^\x20-\x7E\x80-\xFF]', '?', text)
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# 3. Test function to verify your fix works
def test_unicode_cleaning():
    """Test function to verify Unicode cleaning works"""
    test_cases = [
        "Revenue: ₹735 million",
        "Price: €100, £80, ¥1000",
        "Temperature: 25°C",
        "Range: 10–50 units",
        # 'Quote: "Hello World"',
        "Quote: \"Hello World\"",
        "Symbol: α, β, γ values",
        "Math: 2×3=6, 10÷2=5",
        "Comparison: x≥10, y≤5, a≠b",
        None,
        "",
        123,
        ["test", "array"]
    ]
    
    print("Testing Unicode cleaning:")
    for i, test in enumerate(test_cases):
        try:
            result = clean_unicode_text(test)
            print(f"{i+1}. Input: {repr(test)} → Output: {repr(result)}")
        except Exception as e:
            print(f"{i+1}. Input: {repr(test)} → Error: {e}")

# 4. Enhanced error handling for your generate_response_report function
# Add this at the beginning of your function after extracting data:

def validate_and_clean_data(data):
    """Validate and clean all data before PDF generation"""
    cleaned_data = {}
    
    # Clean all text fields
    text_fields = ['question', 'answer', 'summary', 'recommendation']
    for field in text_fields:
        value = data.get(field, '')
        if value and value != 'N/A':
            cleaned_data[field] = clean_unicode_text(value)
        else:
            cleaned_data[field] = value
    
    # Handle rows data
    rows = data.get('rows', [])
    if rows and isinstance(rows, list):
        cleaned_rows = []
        for row in rows:
            if isinstance(row, dict):
                cleaned_row = {}
                for key, value in row.items():
                    cleaned_row[clean_unicode_text(str(key))] = clean_unicode_text(str(value))
                cleaned_rows.append(cleaned_row)
            else:
                cleaned_rows.append(clean_unicode_text(str(row)))
        cleaned_data['rows'] = cleaned_rows
    else:
        cleaned_data['rows'] = rows
    
    # Keep other fields as is
    cleaned_data['chart_base64'] = data.get('chart_base64')
    cleaned_data['chart_config'] = data.get('chart_config')
    
    return cleaned_data


@csrf_exempt
def generate_response_report(request):
    temp_files = []  # Keep track of temporary files for cleanup
    
    try:
        data = json.loads(request.body)
        
        # Extract data
        question = data.get("question", "No Question")
        answer = data.get("answer", "No Answer")
        summary = data.get("summary", "N/A")
        recommendation = data.get("recommendation", "N/A")
        rows = data.get("rows", [])
        chart_base64 = data.get("chart_base64", None)
        
        # Create DataFrame
        df = pd.DataFrame(rows) if rows else pd.DataFrame()
        
        # Initialize PDF
        pdf = ReportPDF()
        pdf.add_page()
        
        # === TITLE SECTION ===
        pdf.set_font(pdf.font_family, "B", 16)
        pdf.safe_cell(0, 15, "Analysis Report", ln=True, align='C')
        pdf.ln(5)
        
        # === QUESTION SECTION ===
        pdf.set_font(pdf.font_family, "B", 12)
        pdf.safe_cell(0, 10, "Question:", ln=True)
        pdf.set_font(pdf.font_family, "", 11)
        wrapped_question = wrap_text(question, width=90)
        pdf.safe_multi_cell(0, 8, wrapped_question)
        pdf.ln(5)
        
        # === ANSWER SECTION ===
        if answer and answer != "No Answer":
            pdf.set_font(pdf.font_family, "B", 12)
            pdf.safe_cell(0, 10, "Answer:", ln=True)
            pdf.set_font(pdf.font_family, "", 11)
            wrapped_answer = wrap_text(answer, width=90)
            pdf.safe_multi_cell(0, 8, wrapped_answer)
            pdf.ln(5)
        
        # === SUMMARY SECTION ===
        if summary and summary != "N/A":
            pdf.set_font(pdf.font_family, "B", 12)
            pdf.safe_cell(0, 10, "Summary:", ln=True)
            pdf.set_font(pdf.font_family, "", 11)
            wrapped_summary = wrap_text(summary, width=90)
            pdf.safe_multi_cell(0, 8, wrapped_summary)
            pdf.ln(5)
        
        # === RECOMMENDATION SECTION ===
        if recommendation and recommendation != "N/A":
            pdf.set_font(pdf.font_family, "B", 12)
            pdf.safe_cell(0, 10, "Recommendation:", ln=True)
            pdf.set_font(pdf.font_family, "", 11)
            wrapped_recommendation = wrap_text(recommendation, width=90)
            pdf.safe_multi_cell(0, 8, wrapped_recommendation)
            pdf.ln(10)
        
        # === CHART SECTION ===
        if chart_base64 and isinstance(chart_base64, str):
            try:
                # Handle different base64 formats
                if "base64," in chart_base64:
                    img_data = chart_base64.split("base64,")[-1]
                else:
                    img_data = chart_base64
                
                # Create temporary file for chart
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                    temp_file.write(base64.b64decode(img_data))
                    chart_path = temp_file.name
                    temp_files.append(chart_path)
                
                # Add chart to PDF
                pdf.set_font(pdf.font_family, "B", 12)
                pdf.safe_cell(0, 10, "Chart:", ln=True)
                pdf.ln(5)
                
                # Calculate image dimensions to fit page
                page_width = pdf.w - 20  # Account for margins
                max_width = min(page_width, 160)
                
                # Get current y position and check if we need a new page
                current_y = pdf.get_y()
                if current_y > 200:  # If too low on page, start new page
                    pdf.add_page()
                
                pdf.image(chart_path, x=10, w=max_width)
                pdf.ln(10)
                
            except Exception as img_err:
                print(f"Chart embedding error: {img_err}")
                pdf.set_font(pdf.font_family, "I", 10)
                pdf.safe_cell(0, 8, "Chart could not be embedded in the report.", ln=True)
                pdf.ln(5)
        
        # === DATA TABLE SECTION ===
        if not df.empty:
            # Start new page for table if needed
            if pdf.get_y() > 150:
                pdf.add_page()
            
            pdf.set_font(pdf.font_family, "B", 12)
            pdf.safe_cell(0, 10, f"Data Table (Top {min(15, len(df))} rows):", ln=True)
            pdf.ln(5)
            
            # Calculate column widths
            num_cols = len(df.columns)
            page_width = pdf.w - 20  # Account for margins
            col_width = min(page_width / num_cols, 35)  # Max 35 units per column
            row_height = 8
            
            # Table headers
            pdf.set_font(pdf.font_family, "B", 9)
            pdf.set_fill_color(200, 220, 255)  # Light blue background
            
            for col in df.columns:
                header_text = truncate_text(str(col), 20)
                pdf.safe_cell(col_width, row_height, header_text, border=1, align='C', fill=True)
            pdf.ln()
            
            # Table data
            pdf.set_font(pdf.font_family, "", 8)
            pdf.set_fill_color(245, 245, 245)  # Light gray for alternate rows
            
            for idx, (_, row) in enumerate(df.head(15).iterrows()):
                fill = idx % 2 == 0  # Alternate row colors
                
                for val in row:
                    cell_text = truncate_text(str(val), 22)
                    pdf.safe_cell(col_width, row_height, cell_text, border=1, align='C', fill=fill, ln=0)
                pdf.ln()
            
            pdf.ln(5)
        
        # === SUMMARY STATISTICS ===
        if not df.empty:
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                pdf.set_font(pdf.font_family, "B", 12)
                pdf.safe_cell(0, 10, "Data Summary:", ln=True)
                pdf.set_font(pdf.font_family, "", 10)
                
                pdf.safe_cell(0, 8, f"Total Rows: {len(df)}", ln=True)
                pdf.safe_cell(0, 8, f"Total Columns: {len(df.columns)}", ln=True)
                pdf.safe_cell(0, 8, f"Numeric Columns: {len(numeric_cols)}", ln=True)
        
        # === GENERATE OUTPUT FILE ===
        # Create unique filename
        safe_question = "".join(c for c in question[:20] if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_question = safe_question.replace(' ', '_')
        unique_id = str(uuid.uuid4())[:8]
        
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        output_filename = f"{output_dir}/report_{safe_question}_{unique_id}.pdf"
        pdf.output(output_filename)
        
        # Create response
        response = FileResponse(
            open(output_filename, "rb"), 
            as_attachment=True, 
            filename="Analysis_Report.pdf"
        )
        
        # Clean up temporary files after a delay (you might want to use a background task)
        import threading
        def cleanup_files():
            import time
            time.sleep(30)  # Wait 30 seconds before cleanup
            try:
                for temp_file in temp_files:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                if os.path.exists(output_filename):
                    os.remove(output_filename)
            except Exception as e:
                print(f"Cleanup error: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_files)
        cleanup_thread.daemon = True
        cleanup_thread.start()
        
        return response
        
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON data"}, status=400)
    except Exception as e:
        # Clean up temporary files on error
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass
        
        import traceback
        print("❌ PDF generation error:", e)
        traceback.print_exc()
    #     return JsonResponse({"error": f"Report generation failed: {str(e)}"}, status=500): 'USD ',  # Dollar (if needed)
    #     '°': ' deg ', # Degree symbol
    #     '–': '-',     # En dash
    #     '—': '-',     # Em dash
    #     ''': "'",     # Left single quotation
    #     ''': "'",     # Right single quotation
    #     '"': '"',     # Left double quotation
    #     '"': '"',     # Right double quotation
    #     '…': '...',   # Ellipsis
    #     '™': '(TM)',  # Trademark
    #     '®': '(R)',   # Registered trademark
    #     '©': '(C)',   # Copyright
    # }
    
    # for unicode_char, replacement in replacements.items():
    #     text = text.replace(unicode_char, replacement)
    
    # # Remove any remaining problematic Unicode characters
    # # Keep only printable ASCII and common extended ASCII
    # text = re.sub(r'[^\x20-\x7E\x80-\xFF]', '?', text)
    
    # return text


def validate_and_clean_data(data):
    """Validate and clean all data before PDF generation"""
    cleaned_data = {}
    
    # Clean all text fields
    text_fields = ['question', 'answer', 'summary', 'recommendation']
    for field in text_fields:
        value = data.get(field, '')
        if value and value != 'N/A':
            cleaned_data[field] = clean_unicode_text(value)
        else:
            cleaned_data[field] = value
    
    # Handle rows data
    rows = data.get('rows', [])
    if rows and isinstance(rows, list):
        cleaned_rows = []
        for row in rows:
            if isinstance(row, dict):
                cleaned_row = {}
                for key, value in row.items():
                    cleaned_row[clean_unicode_text(str(key))] = clean_unicode_text(str(value))
                cleaned_rows.append(cleaned_row)
            else:
                cleaned_rows.append(clean_unicode_text(str(row)))
        cleaned_data['rows'] = cleaned_rows
    else:
        cleaned_data['rows'] = rows
    
    # Keep other fields as is
    cleaned_data['chart_base64'] = data.get('chart_base64')
    cleaned_data['chart_config'] = data.get('chart_config')
    
    return cleaned_data

def debug_unicode_issues(text):
    """Debug function to identify problematic Unicode characters"""
    if not isinstance(text, str):
        text = str(text)
    
    problematic_chars = []
    for i, char in enumerate(text):
        if ord(char) > 255:  # Characters outside Latin-1 range
            problematic_chars.append((i, char, ord(char), hex(ord(char))))
    
    if problematic_chars:
        print("Found problematic Unicode characters:")
        for pos, char, code, hex_code in problematic_chars:
            print(f"  Position {pos}: '{char}' (Unicode: {code}, Hex: {hex_code})")
    else:
        print("No problematic Unicode characters found.")
    
    return problematic_chars

def wrap_text(text, width=100):
    """Wrap text to specified width with Unicode cleaning"""
    if not isinstance(text, str):
        text = str(text)
    text = clean_unicode_text(text)
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=True))

def truncate_text(text, max_length=25):
    """Truncate text for table cells with Unicode cleaning"""
    if not isinstance(text, str):
        text = str(text)
    text = clean_unicode_text(text)
    return text[:max_length-3] + "..." if len(text) > max_length else text

class ReportPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        # Set encoding to handle extended characters better
        try:
            self.add_font('DejaVu', '', 'DejaVuSansCondensed.ttf', uni=True)
            self.font_family = 'DejaVu'
        except:
            # Fallback to Arial with latin-1 encoding
            self.font_family = 'Arial'
    
    def header(self):
        """Add header to each page"""
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Prowesstics AI Analytics Report', 0, 1, 'C')
        self.ln(5)
    
    def footer(self):
        """Add footer to each page"""
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

@csrf_exempt
def generate_response_report(request):
    temp_files = []  # Keep track of temporary files for cleanup
    
    try:
        data = json.loads(request.body)
        
        # Extract data
        question = data.get("question", "No Question")
        answer = data.get("answer", "No Answer")
        summary = data.get("summary", "N/A")
        recommendation = data.get("recommendation", "N/A")
        rows = data.get("rows", [])
        chart_base64 = data.get("chart_base64", None)
        
        # Create DataFrame
        df = pd.DataFrame(rows) if rows else pd.DataFrame()
        
        # Initialize PDF
        pdf = ReportPDF()
        pdf.add_page()
        
        # === TITLE SECTION ===
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 15, "Analysis Report", ln=True, align='C')
        pdf.ln(5)
        
        # === QUESTION SECTION ===
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Question:", ln=True)
        pdf.set_font("Arial", "", 11)
        wrapped_question = wrap_text(question, width=90)
        pdf.multi_cell(0, 8, wrapped_question)
        pdf.ln(5)
        
        # === ANSWER SECTION ===
        if answer and answer != "No Answer":
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Answer:", ln=True)
            pdf.set_font("Arial", "", 11)
            wrapped_answer = wrap_text(answer, width=90)
            pdf.multi_cell(0, 8, wrapped_answer)
            pdf.ln(5)
        
        # === SUMMARY SECTION ===
        if summary and summary != "N/A":
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Summary:", ln=True)
            pdf.set_font("Arial", "", 11)
            wrapped_summary = wrap_text(summary, width=90)
            pdf.multi_cell(0, 8, wrapped_summary)
            pdf.ln(5)
        
        # === RECOMMENDATION SECTION ===
        if recommendation and recommendation != "N/A":
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Recommendation:", ln=True)
            pdf.set_font("Arial", "", 11)
            wrapped_recommendation = wrap_text(recommendation, width=90)
            pdf.multi_cell(0, 8, wrapped_recommendation)
            pdf.ln(10)
        
        # === CHART SECTION ===
        if chart_base64 and isinstance(chart_base64, str):
            try:
                # Handle different base64 formats
                if "base64," in chart_base64:
                    img_data = chart_base64.split("base64,")[-1]
                else:
                    img_data = chart_base64
                
                # Create temporary file for chart
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                    temp_file.write(base64.b64decode(img_data))
                    chart_path = temp_file.name
                    temp_files.append(chart_path)
                
                # Add chart to PDF
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, "Chart:", ln=True)
                pdf.ln(5)
                
                # Calculate image dimensions to fit page
                page_width = pdf.w - 20  # Account for margins
                max_width = min(page_width, 160)
                
                # Get current y position and check if we need a new page
                current_y = pdf.get_y()
                if current_y > 200:  # If too low on page, start new page
                    pdf.add_page()
                
                pdf.image(chart_path, x=10, w=max_width)
                pdf.ln(10)
                
            except Exception as img_err:
                print(f"Chart embedding error: {img_err}")
                pdf.set_font("Arial", "I", 10)
                pdf.cell(0, 8, "⚠️ Chart could not be embedded in the report.", ln=True)
                pdf.ln(5)
        
        # === DATA TABLE SECTION ===
        if not df.empty:
            # Start new page for table if needed
            if pdf.get_y() > 150:
                pdf.add_page()
            
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, f"Data Table (Top {min(15, len(df))} rows):", ln=True)
            pdf.ln(5)
            
            # Calculate column widths
            num_cols = len(df.columns)
            page_width = pdf.w - 20  # Account for margins
            col_width = min(page_width / num_cols, 35)  # Max 35 units per column
            row_height = 8
            
            # Table headers
            pdf.set_font("Arial", "B", 9)
            pdf.set_fill_color(200, 220, 255)  # Light blue background
            
            for col in df.columns:
                header_text = truncate_text(str(col), 20)
                pdf.cell(col_width, row_height, header_text, border=1, align='C', fill=True)
            pdf.ln()
            
            # Table data
            pdf.set_font("Arial", "", 8)
            pdf.set_fill_color(245, 245, 245)  # Light gray for alternate rows
            
            for idx, (_, row) in enumerate(df.head(15).iterrows()):
                fill = idx % 2 == 0  # Alternate row colors
                
                for val in row:
                    cell_text = truncate_text(str(val), 22)
                    pdf.cell(col_width, row_height, cell_text, border=1, align='C', fill=fill, ln=0)
                pdf.ln()
            
            pdf.ln(5)
        
        # === SUMMARY STATISTICS ===
        if not df.empty:
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, "Data Summary:", ln=True)
                pdf.set_font("Arial", "", 10)
                
                pdf.cell(0, 8, f"Total Rows: {len(df)}", ln=True)
                pdf.cell(0, 8, f"Total Columns: {len(df.columns)}", ln=True)
                pdf.cell(0, 8, f"Numeric Columns: {len(numeric_cols)}", ln=True)
        
        # === GENERATE OUTPUT FILE ===
        # Create unique filename
        safe_question = "".join(c for c in question[:20] if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_question = safe_question.replace(' ', '_')
        unique_id = str(uuid.uuid4())[:8]
        
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        output_filename = f"{output_dir}/report_{safe_question}_{unique_id}.pdf"
        pdf.output(output_filename)
        
        # Create response
        response = FileResponse(
            open(output_filename, "rb"), 
            as_attachment=True, 
            filename="Analysis_Report.pdf"
        )
        
        # Clean up temporary files after a delay (you might want to use a background task)
        import threading
        def cleanup_files():
            import time
            time.sleep(30)  # Wait 30 seconds before cleanup
            try:
                for temp_file in temp_files:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                if os.path.exists(output_filename):
                    os.remove(output_filename)
            except Exception as e:
                print(f"Cleanup error: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_files)
        cleanup_thread.daemon = True
        cleanup_thread.start()
        
        return response
        
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON data"}, status=400)
    except Exception as e:
        # Clean up temporary files on error
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass
        
        import traceback
        print("❌ PDF generation error:", e)
        traceback.print_exc()
        return JsonResponse({"error": f"Report generation failed: {str(e)}"}, status=500)