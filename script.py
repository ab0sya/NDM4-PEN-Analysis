import prody as pr
import networkx as nx
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Suppress ProDy's terminal spam so we only see our results
pr.confProDy(verbosity='error')

def calculate_pen_centrality(pdb_id):
    print(f"Processing {pdb_id}...")
    
    # 1. Fetch and Parse the PDB file directly from the RCSB database
    protein = pr.parsePDB(pdb_id)
    
    # 2. The Critical Selection Step
    # We isolate Chain A. We keep C-alpha (CA) OR Zinc (ZN).
    selection = protein.select('chain A and ((protein and name CA) or resname ZN)')
    
    if selection is None:
        raise ValueError(f"Could not extract CA and ZN from {pdb_id}")
        
    # 3. Build the Anisotropic Network Model (ANM)
    anm = pr.ANM(pdb_id)
    anm.buildHessian(selection)
    anm.calcModes() 
    
    # 4. Generate the Cross-Correlation Matrix
    # This matrix tells us how much residue 'i' moves when residue 'j' moves
    cc_matrix = pr.calcCrossCorr(anm)
    
    # 5. Build the Protein Energy Network (PEN)
    G = nx.Graph()
    num_nodes = cc_matrix.shape[0]
    
    resnums = selection.getResnums()
    resnames = selection.getResnames()
    
    # Add nodes to the graph
    for i in range(num_nodes):
        G.add_node(i, resnum=resnums[i], resname=resnames[i])
        
    # Add weighted edges based on absolute cross-correlation
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            weight = abs(cc_matrix[i, j])
            if weight > 0.01: # Filter out microscopic mathematical noise
                G.add_edge(i, j, weight=weight)
                
    # 6. Calculate Eigenvector Centrality
    centrality = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
    
    # 7. Convert to Z-scores for statistical significance
    cent_values = list(centrality.values())
    z_scores = stats.zscore(cent_values)
    
    # Map results to readable biological IDs (e.g., "MET154", "ZN1")
    results = {}
    for i in range(num_nodes):
        res_id = f"{resnames[i]}{resnums[i]}"
        results[res_id] = z_scores[i]
        
    return results

def main():
    wt_pdb = '3spu'
    mut_pdb = '8sk2'
    
    wt_results = calculate_pen_centrality(wt_pdb)
    mut_results = calculate_pen_centrality(mut_pdb)
    
    print("\n--- CALCULATING DELTA Z (Mutant - Wild Type) ---")
    
    delta_z = {}
    for res_id in wt_results.keys():
        if res_id in mut_results:
            delta_z[res_id] = mut_results[res_id] - wt_results[res_id]
            
    sorted_delta = sorted(delta_z.items(), key=lambda item: item[1], reverse=True)
    
    print("\nTop 20 Hubs that GAINED Centrality in the Mutant:")
    for res_id, dz in sorted_delta[:20]:
        wt_val = wt_results[res_id]
        mut_val = mut_results[res_id]
        print(f"{res_id}: +{dz:.3f} Z  (Trajectory: {wt_val:.2f} -> {mut_val:.2f})")
        
    print("\nTop 20 Hubs that LOST Centrality in the Mutant:")
    for res_id, dz in reversed(sorted_delta[-20:]):
        wt_val = wt_results[res_id]
        mut_val = mut_results[res_id]
        print(f"{res_id}: {dz:.3f} Z  (Trajectory: {wt_val:.2f} -> {mut_val:.2f})")

if __name__ == "__main__":
    main()