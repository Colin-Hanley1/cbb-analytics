import pandas as pd
import subprocess
import datetime
import os

# --- CONFIGURATION ---
# The names of your scripts. ensure these match your actual filenames!
SCRAPER_SCRIPT = "scraper.py" 
RATINGS_SCRIPT = "adj.py"
OUTPUT_HTML = "index.html"

def run_script(script_name):
    print(f"--- Running {script_name} ---")
    try:
        # Run the script and wait for it to finish
        subprocess.run(["python3", script_name], check=True)
        print(f"Successfully ran {script_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        # We don't exit here because we might still want to publish old data if scraper fails
        pass

def generate_html():
    print("Generating Website...")
    
    # 1. Load the Ratings Data
    try:
        df = pd.read_csv("cbb_ratings.csv")
    except FileNotFoundError:
        print("Error: cbb_ratings.csv not found. Did the ratings script fail?")
        return

    # 2. Load the HTML Template
    try:
        with open("template.html", "r") as f:
            template = f.read()
    except FileNotFoundError:
        print("Error: template.html not found.")
        return

    # 3. Convert Dataframe to HTML Table Rows
    table_rows = ""
    for _, row in df.iterrows():
        # Color logic: Green for positive, Red for negative
        em_color = "text-green-600" if row['Blended_AdjEM'] > 0 else "text-red-600"
        
        # Helper for formatting numbers
        def fmt(val, decimals=1):
            try:
                return f"{float(val):.{decimals}f}"
            except:
                return str(val)

        row_html = f"""
        <tr class="hover:bg-gray-50 border-b border-gray-100 transition-colors">
            <td class="px-4 py-3 font-bold text-gray-700 text-center">{int(row['Rank'])}</td>
            <td class="px-4 py-3 text-gray-900 font-medium">{row['Team']}</td>
            <td class="px-4 py-3 text-gray-600 text-center font-mono text-xs">{row.get('W-L', '-')}</td>
            <td class="px-4 py-3 font-bold {em_color} text-right">{fmt(row['Blended_AdjEM'], 2)}</td>
            <td class="px-4 py-3 text-gray-700 text-right">{fmt(row['AdjO'])}</td>
            <td class="px-4 py-3 text-gray-700 text-right">{fmt(row['AdjD'])}</td>
            <td class="px-4 py-3 text-gray-700 text-right">{fmt(row['AdjT'])}</td>
            <td class="px-4 py-3 text-gray-500 text-right text-xs">{fmt(row.get('SOS', 0.0), 2)}</td>
        </tr>
        """
        table_rows += row_html

    # 4. Inject Data into Template
    # We get current time in UTC
    now = datetime.datetime.now(datetime.timezone.utc)
    last_updated = now.strftime("%Y-%m-%d %H:%M UTC")
    
    final_html = template.replace("{{TABLE_ROWS}}", table_rows)
    final_html = final_html.replace("{{LAST_UPDATED}}", last_updated)

    # 5. Save the final index.html
    with open(OUTPUT_HTML, "w") as f:
        f.write(final_html)
    
    print(f"Website generated at {OUTPUT_HTML}")

if __name__ == "__main__":
    # Step 1: Update the Data (Scrape new games)
    run_script(SCRAPER_SCRIPT)
    
    # Step 2: Recalculate Ratings (Run the math)
    run_script(RATINGS_SCRIPT)
    
    # Step 3: Build the Website (Generate HTML)
    generate_html()