from shiny import App, render, ui, reactive
import repo4eu
import networkx as nx
import io
import base64
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import time
import asyncio

model, nodes, data, G = repo4eu.load_model('model_version_3.1_mashup.pth')



app_ui = ui.page_fluid(
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
            
        	ui.nav("Explanations", 
            ui.input_text("diso_2", "Enter disorder (you should introduce the MONDO ID in lowercase, ie. mondo.0005015):", placeholder='text'),
            ui.input_text("drug", "Enter disorder (you should introduce the DB ID in lowercase, ie. drugbank.DB09043):", placeholder='text'),
            ui.input_action_button("go_2", "Update table"),
            ui.output_table("explanations_scores"), 
            ),
    
    
    		ui.nav("Plot Explanation", 
            ui.input_text("exp_id", "Enter the ID of the explanation you would like to plot", placeholder='text'),
            ui.input_action_button("go_3", "Update plot"),
            ui.output_image("plot_explanation"), 
            ),
    ),
)
    



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
       	    return print(diso) 
       	    

    @output
    @render.table
    @reactive.event(input.go_2, ignore_none=False)
    def explanations_scores():
        global path_list, scores
    
        diso = input.diso_2()
        drug = input.drug()
        if diso in G.nodes() and drug in G.nodes():
        	path_list, scores = repo4eu.best_explanations(G, nodes, model, drug, diso, 3)
        	return scores[:10]
        else: 	
        	return print(drug, diso)
        	

    @output
    @render.image(delete_file=True)
    @reactive.event(input.go_3, ignore_none=False)
    def plot_explanation():
        global path_list, scores
        try: 
            plt.clf()
            edge_labels = {}
            exp_id = input.exp_id()
            best_G = path_list[int(exp_id)]

            edges_dic = nx.get_edge_attributes(best_G, 'edge_name')
            edge_labels = {edge: edges_dic[edge] for edge in best_G.edges() if edge in edges_dic}

            # Adjusted for correct node names
            node_types = dict(zip(nodes['Nodes Name'], nodes['Category']))
            unique_types = list(set(node_types.values()))
            colors = cm.tab10.colors
            type_colors = {node_type: colors[i % len(colors)] for i, node_type in enumerate(unique_types)}

            node_colors = [type_colors.get(node_types.get(node, None), 'lightgray') for node in best_G.nodes()]
            pos = nx.shell_layout(best_G)
            index_map = dict(zip(nodes['Nodes Name'], nodes['Display Name']))
            renamed_labels = {node: index_map.get(node, node) for node in best_G.nodes()}

            # Draw the graph with edge labels
            
            nx.draw(best_G, pos, labels=renamed_labels, with_labels=True, node_size=500, node_color=node_colors, font_weight='bold', font_size=6)
            # nx.draw(best_G, pos, with_labels=True, node_size=500, node_color='lightblue', font_weight='bold')
            nx.draw_networkx_edge_labels(best_G, pos, edge_labels=edge_labels)

            # Create a legend
            legend_patches = [mpatches.Patch(color=color, label=category) for category, color in type_colors.items()]
            plt.legend(handles=legend_patches, loc='upper left')

            plt.savefig('./image.png', format='png')
            img: ImgData = {"src": './image.png', "width": "1000px"}
        
            return img
            
        except Exception as e:
            print(f"Error occurred: {e}")
            return None
            
    @output
    @render.text
    @reactive.event(input.go)
    async def compute():
        with ui.Progress(min=1, max=15) as p:
            p.set(message="Calculation in progress", detail="This may take a while...")

            for i in range(1, 15):
                p.set(i, message="Computing")
                await asyncio.sleep(0.1)
                # Normally use time.sleep() instead, but it doesn't yet work in Pyodide.
                # https://github.com/pyodide/pyodide/issues/2354

        return "Done computing!"

app = App(app_ui, server)
