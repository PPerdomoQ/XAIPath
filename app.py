from shiny import App, render, ui, reactive
import repo4eu
import networkx as nx
import matplotlib.cm as cm
import asyncio
import html as html_lib
import re
from pyvis.network import Network
import urllib.parse

model, nodes, data, G = repo4eu.load_model('model_version_3.1_mashup.pth')

# Global variables for cross-function state
path_list = None
scores = None

app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.style("""
            /* Spinner Styles */
            .spinner {
                border: 4px solid rgba(0, 0, 0, 0.1);
                width: 36px;
                height: 36px;
                border-radius: 50%;
                border-left-color: #09f;
                animation: spin 1s linear infinite;
                display: inline-block;
                vertical-align: middle;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .loading-container {
                display: none; /* Hidden by default */
                align-items: center;
                justify-content: center;
                padding: 15px;
                margin: 10px 0;
                background-color: #f8f9fa;
                border-radius: 8px;
                border: 1px solid #e9ecef;
                color: #495057;
                font-weight: 500;
                gap: 10px;
            }
        """),
        ui.tags.script("""
            $(document).on('click', '#go_2', function() {
                $('#loading-spinner-container').css('display', 'flex');
            });
            
            $(document).on('shiny:value', function(event) {
                if (event.name === 'interactive_graph') {
                    $('#loading-spinner-container').hide();
                }
            });
            
            $(document).on('shiny:error', function(event) {
                if (event.name === 'interactive_graph') {
                    $('#loading-spinner-container').hide();
                }
            });
            
            Shiny.addCustomMessageHandler('trigger_plot', function(message) {
                // Wait a bit to ensure Shiny inputs are updated before clicking
                setTimeout(function() {
                    $('#go_2').click();
                }, 500);
            });
        """)
    ),
    ui.navset_tab(
        ui.nav("Predictions", 
            ui.input_text("diso", "Enter disorder (you should introduce the MONDO ID in lowercase, ie. mondo.0005015):", placeholder='text'),
            ui.input_slider("k", "Longest shorest distance between drug and disease. The higher this value, tha further away you will allow the drug to be from the disease.", min=2, max=5, value=2),
            ui.input_action_button("go", "Update table"),
            ui.panel_conditional(
        "input.go", ui.tags.div('Loading ... ')
    ),
            ui.output_table("summary_data"), 
            ),
            
        ui.nav(
            "Explanations",
            ui.row(
                ui.column(4,
                    ui.input_text(
                        "diso_2",
                        "Enter disorder (MONDO ID):",
                        placeholder='mondo.0005015'
                    )
                ),
                ui.column(4,
                    ui.input_text(
                        "drug",
                        "Enter drug (DB ID):",
                        placeholder='drugbank.DB09043'
                    )
                ),
                ui.column(4,
                    ui.input_slider("k_2", "Search depth (max hops):", min=2, max=5, value=3),
                )
            ),
            ui.row(
                ui.column(8,
                    ui.input_checkbox(
                        "use_minhash",
                        "Use boosted MinHash explanations",
                        value=False,
                    ),
                    ui.input_action_button("go_2", "Plot Explanations", class_="btn-primary"),
                ),
                ui.column(4,
                    ui.input_select(
                        "plot_mode",
                        "Select view:",
                        choices={
                            "merged": "Top 5 (Merged)",
                            "0": "Path 1",
                            "1": "Path 2",
                            "2": "Path 3",
                            "3": "Path 4",
                            "4": "Path 5"
                        }
                    )
                )
            ),
            ui.tags.div(
                ui.div(class_="spinner"),
                ui.span("Calculating and plotting... Please wait."),
                id="loading-spinner-container",
                class_="loading-container"
            ),
            ui.tags.div(
                ui.output_ui("interactive_graph"),
                style="height: 750px; width: 100%; border: 1px solid #ddd; border-radius: 4px; overflow: hidden; margin-top: 15px;"
            ),
        ),
    ),
)
    

def get_pyvis_utils_js():
    """Returns the hardcoded content of pyvis's utils.js to ensure reliable injection."""
    return """
function drawGraph() {
    for (var i = 0; i < network.body.data.nodes.length; i++) {
        network.body.data.nodes._data[i].hidden = false;
    }
    for (var i = 0; i < network.body.data.edges.length; i++) {
        network.body.data.edges._data[i].hidden = false;
    }
    network.redraw();
}

function findNode(id) {
    return network.body.data.nodes._data[id];
}
"""

def server(input, output, session):

    @output
    @render.table
    @reactive.event(input.go, ignore_none=False)
    def summary_data(): 
        # Get the value of the input field 'diso'
        diso = input.diso()
        k = input.k()

        # Check if the 'diso' value is not empty before fetching candidates
        if diso in G.nodes():
            df = repo4eu.get_candidates(model, nodes, data, G, diso, k = k)[:10]
            return df
        else: 
            return None 

    # URL Deep Linking Logic
    url_loaded = reactive.Value(False)
    
    @reactive.Effect
    async def _():
        if url_loaded():
            return
            
        # Get query parameters from the URL
        search = session.input[".clientdata_url_search"]()
        if not search:
            return
            
        params = urllib.parse.parse_qs(search.lstrip('?'))
        drug = params.get("drug", [None])[0]
        diso = params.get("diso", [None])[0]
        
        if drug and diso:
            print(f"Deep link detected: Drug={drug}, Disorder={diso}", flush=True)
            # Update inputs
            ui.update_text("drug", value=drug)
            ui.update_text("diso_2", value=diso)
            ui.update_slider("k_2", value=3)
            ui.update_checkbox("use_minhash", value=True)
            
            # Mark as loaded so we don't re-trigger on other reactive updates
            url_loaded.set(True)
            
            # Trigger the JS click event to show spinner and start computation
            await session.send_custom_message("trigger_plot", {})

    # Calculated paths storage
    calculated_paths = reactive.Value(None)

    @reactive.Effect
    @reactive.event(input.go_2)
    async def compute_explanations():
        try:
            diso = input.diso_2().strip()
            drug = input.drug().strip()
            k = input.k_2()
            use_minhash = input.use_minhash()

            print(f"Computing explanations for: Drug={drug}, Disorder={diso}, Depth={k}")
            
            if diso in G.nodes() and drug in G.nodes():
                if use_minhash:
                    # Run CPU-intensive task in a separate thread to keep UI responsive
                    paths, _ = await asyncio.to_thread(
                        repo4eu.best_explanations_minhash, G.copy(), nodes, model, drug, diso, k
                    )
                else:
                    paths, _ = await asyncio.to_thread(
                        repo4eu.best_explanations, G, nodes, model, drug, diso, k
                    )
                print(f"Found {len(paths)} paths.", flush=True)
                calculated_paths.set(paths)
            else:
                errors = []
                if diso not in G.nodes():
                    errors.append(f"Disorder '{diso}' not found.")
                if drug not in G.nodes():
                    errors.append(f"Drug '{drug}' not found.")
                
                error_msg = " ".join(errors)
                print(error_msg, flush=True)
                calculated_paths.set(f"Error: {error_msg}")
        except Exception as e:
            print(f"Error in compute_explanations: {e}", flush=True)
            calculated_paths.set(f"Error: Internal calculation error.")

    @output
    @render.ui
    def interactive_graph():
        val = calculated_paths()
        try:
            if val is None:
                return ui.HTML('<p style="padding:20px; color:#888;">Enter disorder/drug and click "Plot Explanations" to begin.</p>')
            
            if isinstance(val, str) and val.startswith("Error:"):
                return ui.HTML(f'<p style="padding:20px; color:red;">{val}</p>')
            
            paths = val
            if not paths:
                return ui.HTML('<p style="padding:20px; color:orange;">No explanation paths found for this pair at the current depth. Try increasing the search depth.</p>')

            mode = input.plot_mode()
            
            # Select the subgraph(s) based on mode
            if mode == "merged":
                top_n = min(5, len(paths))
                best_G = nx.compose_all([paths[i] for i in range(top_n)])
                view_title = f"Top {top_n} Merged Explanations"
            else:
                idx = int(mode)
                if idx >= len(paths):
                    return ui.HTML(f'<p style="padding:20px;">Explanation path {idx+1} not found.</p>')
                best_G = paths[idx]
                view_title = f"Explanation Path {idx+1}"

            print(f"Plotting: {view_title} with {best_G.number_of_nodes()} nodes and {best_G.number_of_edges()} edges", flush=True)

            if best_G.number_of_nodes() == 0:
                return ui.HTML('<p style="padding:20px; color:orange;">The selected subgraph has no nodes to plot.</p>')

            # Build colour map keyed on category for nodes
            node_types = dict(zip(nodes['Nodes Name'], nodes['Category']))
            unique_types = list(set(node_types.values()))
            mpl_colors = cm.tab10.colors
            type_colors = {
                node_type: '#{:02x}{:02x}{:02x}'.format(
                    int(mpl_colors[i % len(mpl_colors)][0] * 255),
                    int(mpl_colors[i % len(mpl_colors)][1] * 255),
                    int(mpl_colors[i % len(mpl_colors)][2] * 255),
                )
                for i, node_type in enumerate(unique_types)
            }

            index_map = dict(zip(nodes['Nodes Name'], nodes['Display Name']))

            net = Network(
                height="680px",
                width="100%",
                bgcolor="#ffffff",
                font_color="#333333",
                directed=False,
                notebook=False,
            )
            # Use remote CDN to avoid 404 on local lib/vis-network.min.js
            try:
                net.set_cdn_resources('remote')
            except AttributeError:
                pass
                
            net.barnes_hut(spring_length=150, spring_strength=0.005, gravity=-20000)

            for node in best_G.nodes():
                category = node_types.get(node, "Unknown")
                color = type_colors.get(category, "#cccccc")
                label = index_map.get(node, node)
                net.add_node(
                    node,
                    label=label,
                    title=f"<b>{label}</b><br/>Category: {category}<br/>ID: {node}",
                    color=color,
                    size=18,
                    font={"size": 12, "face": "arial"},
                )

            edges_dic = nx.get_edge_attributes(best_G, 'edge_name')
            for u, v in best_G.edges():
                edge_label = edges_dic.get((u, v), edges_dic.get((v, u), ""))
                net.add_edge(
                    u, v, 
                    label=edge_label, 
                    title=edge_label,
                    color="#888888",
                    width=2,
                    font={"size": 10}
                )

            active_categories = {node_types.get(n) for n in best_G.nodes()}
            cat_items = "".join(
                f'<span style="display:inline-flex;align-items:center;margin-right:14px;">'
                f'<span style="width:12px;height:12px;border-radius:2px;background:{color};'
                f'display:inline-block;margin-right:5px;"></span>'
                f'<span style="font-size:11px;font-family:arial;">{cat}</span></span>'
                for cat, color in type_colors.items()
                if cat in active_categories
            )
            
            legend_bar = (
                f'<div style="padding:8px 10px;background:#fcfcfc;border-bottom:1px solid #eee;'
                f'font-family:arial;display:flex;align-items:center;justify-content:space-between;">'
                f'<div><b>{view_title}</b></div>'
                f'<div style="display:flex;flex-wrap:wrap;"><b>Nodes:</b> &nbsp; {cat_items}</div>'
                f'</div>'
            )

            html_str = net.generate_html(notebook=False)
            
            # 1. Strip ALL local library references to prevent 404s
            # Pyvis generates <script src="lib/..."> or <link href="lib/...">
            html_str = re.sub(r'<script [^>]*src=["\']lib/[^>]*></script>', '', html_str)
            html_str = re.sub(r'<link [^>]*href=["\']lib/[^>]*>', '', html_str)
            
            # 2. Inject the necessary remote scripts into <head>
            cdn_scripts = """
            <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.9/standalone/umd/vis-network.min.js"></script>
            <script type="text/javascript">
            """ + get_pyvis_utils_js() + """
            </script>
            """
            
            if "</head>" in html_str:
                html_str = html_str.replace("</head>", cdn_scripts + "</head>")
                print("SUCCESS: Injected CDN and hardcoded scripts into HTML head.")
            else:
                # Fallback if no head tag found
                html_str = cdn_scripts + html_str
                print("WARNING: No </head> found, prepending scripts to HTML.")

            escaped = html_lib.escape(html_str, quote=True)

            iframe = (
                f'<iframe srcdoc="{escaped}" '
                f'style="width:100%;height:650px;border:none;" '
                f'sandbox="allow-scripts"></iframe>'
            )

            return ui.HTML(
                f'<div style="display:flex;flex-direction:column;height:700px;">'
                f'{legend_bar}{iframe}</div>'
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            return ui.HTML(f'<p style="color:red; padding:20px;">Error rendering graph: {e}</p>')
            
    @output
    @render.text
    @reactive.event(input.go)
    async def compute():
        with ui.Progress(min=1, max=15) as p:
            p.set(message="Calculation in progress", detail="This may take a while...")

            for i in range(1, 15):
                p.set(i, message="Computing")
                await asyncio.sleep(0.1)

        return "Done computing!"

app = App(app_ui, server)

