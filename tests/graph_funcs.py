import networkx as nx
import plotly.graph_objects as go
import math
import numpy as np
from ast import literal_eval
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data


def create_graph(base_peptide,branch_points,branches):
    """
    Creates graph of branched peptides. Returns atoms, edges, atom properties.

    """
    if isinstance(branch_points, str):
        branch_points = literal_eval(branch_points)
        branches = branches.split(',')
    
    edges = []
    atoms = {}

    current_node = 0
    for i, atom in enumerate(base_peptide):
        atoms[current_node] = atom
        if i>0:
            edges.append((current_node-1, current_node))
        current_node += 1

    if isinstance(branch_points, tuple):
        for j, bp in enumerate(branch_points):
            start_node = bp
            for atom in branches[j]:
                atoms[current_node] = atom
                edges.append((start_node,current_node))
                start_node = current_node
                current_node += 1
    
    elif math.isnan(branch_points) == False:
        start_node = branch_points
        for atom in branches:
            atoms[current_node] = atom
            edges.append((start_node,current_node))
            start_node = current_node
            current_node += 1
    
    edges_r = []
    for edge in edges:
        edges_r.append((edge[1],edge[0]))

    edges.extend(edges_r)

    atoms_features = np.zeros((len(atoms),4))   
    for item in atoms:
        atom = atoms[item]
        if atom == 'A':
            atoms_features[item,0] = 1
        if atom == 'H':
            atoms_features[item,1] = 1
        if atom == 'N':
            atoms_features[item,2] = 1
        if atom == 'K':
            atoms_features[item,3] = 1

    return(atoms, edges, atoms_features)

def add_branch(atoms,edges,start_node, branch):
    """
    Add branch to existing graph. Requires explicit setting of starting node. 
    """
    current_node = len(atoms)
    for atom in branch:
        atoms[current_node] = atom
        edges.append((start_node, current_node))
        edges.append((current_node,start_node))
        start_node = current_node
        current_node += 1

    atoms_features = np.zeros((len(atoms),4))   
    for item in atoms:
        atom = atoms[item]
        if atom == 'A':
            atoms_features[item,0] = 1
        if atom == 'H':
            atoms_features[item,1] = 1
        if atom == 'N':
            atoms_features[item,2] = 1
        if atom == 'K':
            atoms_features[item,3] = 1
    
    return(atoms, edges, atoms_features)


def plotly_viz(atoms,edges, title='peptide'):
    G = nx.Graph()
    G.add_edges_from(edges)

    pos = nx.planar_layout(G)
    pos = nx.spring_layout(G, pos=pos)
    
    x_pos = [pos[i][0] for i in range(len(pos))]
    y_pos = [pos[i][1] for i in range(len(pos))]

    edge_x = []
    edge_y = []
    for edge in edges:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='gray'),
        hoverinfo='none',
        mode='lines'
    )


    node_labels = list(atoms.items())

    colours_dict = {'A':'#9f666d',
                    'H':'#3c9cb4',
                    'K':'#f9c107',
                    'N':'#656f72'}
    colour_list = []
    for atom in list(atoms.values()):
        colour_list.append(colours_dict[atom])

    node_trace = go.Scatter(
        x=x_pos, y=y_pos,
        mode='markers+text',
        text=node_labels,
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color=colour_list,
            size=20,
            line_width=2
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=50),
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False)
                    ))

    fig.update_layout(title=title)

    fig.show()

def plt_vis(atoms,edges, edge_mask=None, draw_edge_labels=False, node_colors=None, title=None, savefig=False, normalize=True):
    G = nx.Graph()
    G.add_edges_from(edges)

    g = G.copy().to_undirected()

    pos = nx.planar_layout(g)
    pos = nx.spring_layout(g, pos=pos)

    if edge_mask is None:
        edge_color = 'black'
        widths = None
    else:
        edge_color = [edge_mask[(u, v)] for u, v in g.edges()]
        widths = [x * 10 for x in edge_color]
    
    if normalize and edge_mask is not None:
        edge_color = [(x - min(edge_color)) / (max(edge_color) - min(edge_color)) for x in edge_color]
        widths = [x * 10 for x in edge_color]


    if node_colors is None:
        node_color = 'azure'
    else:
        colours_dict = {'A':'#9f666d',
                    'H':'#3c9cb4',
                    'K':'#f9c107',
                    'N':'#656f72'}
        node_color = []
        for atom in list(atoms.values()):
            node_color.append(colours_dict[atom])

    nx.draw(g, pos=pos, labels=atoms, width=widths,
            edge_color=edge_color, edge_cmap=plt.cm.Blues,
            node_color=node_color)
    
    if draw_edge_labels and edge_mask is not None:
        edge_labels = {k: ('%.2f' % v) for k, v in edge_mask.items()}    
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels,
                                    font_color='red')
        
    if title is not None:
        plt.title(title, fontsize=20)

    if savefig:
        plt.savefig(f'./{title}.pdf')
    plt.show()


def create_dataset(df):
    dataset = []
    for row in range(0,len(df)):
        try:
            flag = np.isnan(df.iloc[row,4])
            atoms, edges, atom_features = create_graph(df.iloc[row,1],df.iloc[row,2],df.iloc[row,3])
        except:
            print(f'Multiple branches for peptide {df.iloc[row,0]}')
            repeat = df['rep'].iloc[row]
            if repeat == 'no':       
                atoms, edges, atom_features = create_graph(df.iloc[row,1],df.iloc[row,2],df.iloc[row,3])
                plotly_viz(atoms,edges, title=f'{df.iloc[row,0]} original')
                num_additional_branches = int(df['ebps'].iloc[row])
                
                for i in range(num_additional_branches):
                    node = int(input("Denote Attached Node: "))
                    branch = df.iloc[row,(df.columns.get_loc("ebps"))+1+i] 
                    atoms, edges, atom_features = add_branch(atoms, edges, node, branch) 
                    plotly_viz(atoms,edges, title=f'{df.iloc[row,0]} updated')

        edge_tensor = torch.tensor(edges,dtype=torch.long)
        atom_tensor = torch.tensor(atom_features)
        features = torch.tensor([(df.iloc[row, 5],df.iloc[row, 6])])
        dataset.append(Data(x=atom_tensor, edge_index=edge_tensor.t().contiguous(),features=features))
    return dataset
