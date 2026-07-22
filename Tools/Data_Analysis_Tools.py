import os
import logging
import pandas as pd
import plotly.express as px
from typing import List, Dict, Any
from langchain_core.tools import tool

# Configure logging
logger = logging.getLogger(__name__)

# Ensure output directories exist for our agent's deliverables
os.makedirs("outputs/charts", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True) 

@tool
def generate_chart(data: List[Dict[str, Any]], chart_type: str, x_column: str, y_column: str, title: str) -> str:
    """
    Generates a chart from a list of data dictionaries and saves it as an interactive HTML file.
    Use this after fetching data via SQL to visualize it for the user.
    
    Args:
        data: The raw list of dictionaries (from execute_sql_query).
        chart_type: Must be one of 'bar', 'line', 'pie', 'scatter'.
        x_column: The dictionary key to use for the X-axis (or labels for pie).
        y_column: The dictionary key to use for the Y-axis (or values for pie).
        title: The title of the chart.
    """
    try:
        if not data:
            return "Error: No data provided to generate chart. Did your SQL query return results?"
            
        # Convert list of dicts to pandas DataFrame
        df = pd.DataFrame(data)
        
        # Verify columns exist
        if x_column not in df.columns or y_column not in df.columns:
            return f"Error: Columns '{x_column}' or '{y_column}' not found in the data. Available columns: {list(df.columns)}"
        
        # Generate the specified chart
        if chart_type.lower() == 'bar':
            fig = px.bar(df, x=x_column, y=y_column, title=title)
        elif chart_type.lower() == 'line':
            fig = px.line(df, x=x_column, y=y_column, title=title)
        elif chart_type.lower() == 'pie':
            fig = px.pie(df, names=x_column, values=y_column, title=title)
        elif chart_type.lower() == 'scatter':
            fig = px.scatter(df, x=x_column, y=y_column, title=title)
        else:
            return f"Error: Unsupported chart type '{chart_type}'. Use bar, line, pie, or scatter."
            
        # Save to disk
        safe_title = title.replace(' ', '_').replace('/', '-').lower()
        filepath = f"outputs/charts/{safe_title}.html"
        fig.write_html(filepath)
        
        return f"SUCCESS: Chart generated and saved to {filepath}. Tell the user the path."
        
    except Exception as e:
        logger.error(f"Chart Generation Error: {str(e)}")
        return f"CHART_ERROR: {str(e)}\nPlease check the data structure."

@tool
def generate_report(title: str, sql_query_used: str, data_summary: str, insights: str) -> str:
    """
    Generates a professional markdown report containing the data summary and analytical insights, 
    and saves it to the disk.
    
    Args:
        title: The title of the report.
        sql_query_used: The exact SQL query that was run to get the data.
        data_summary: A brief description of the raw data returned.
        insights: The analytical conclusions drawn from the data.
    """
    try:
        safe_title = title.replace(' ', '_').replace('/', '-').lower()
        filepath = f"outputs/reports/{safe_title}.md"
        
        # Build Markdown structure
        report_content = f"# {title}\n\n"
        report_content += f"## SQL Query Used\n```sql\n{sql_query_used}\n```\n\n"
        report_content += f"## Data Summary\n{data_summary}\n\n"
        report_content += f"## Analytical Insights\n{insights}\n"
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report_content)
            
        return f"SUCCESS: Report generated and saved to {filepath}. Tell the user the path."
        
    except Exception as e:
        logger.error(f"Report Generation Error: {str(e)}")
        return f"REPORT_ERROR: {str(e)}"