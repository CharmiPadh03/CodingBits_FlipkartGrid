import json, sys, os
sys.stdout.reconfigure(line_buffering=True)
os.chdir(r"E:\Flipkart GridLock Hackathon 2.0\H3Uber")

with open('patrol_forecast_v3.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']

for i, cell in enumerate(code_cells):
    src = ''.join(cell['source'])
    if 'IFrame(' in src:
        src = src[:src.index('\nfrom IPython')] + "\n"
    src = src.replace('plt.show()', '')

    print(f"\n{'='*55}\n--- cell {i} ---\n{'='*55}")
    try:
        exec(compile(src, f'<cell {i}>', 'exec'), globals())
    except Exception as e:
        print(f"ERROR cell {i}: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

print("\n\nALL DONE")
