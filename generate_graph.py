import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import argparse
import re  # Importando regex
from pathlib import Path

def parse_counts(json_str):
    """
    Lê o JSON e extrai apenas AA (MHC) e CC (Peptídeo).
    Ignora AC e CA.
    """
    try:
        if '""' in json_str:
            json_str = json_str.replace('""', '"')
        json_str = json_str.strip('"')
        data = json.loads(json_str)
        
        mhc_found = data.get("AA", 0)
        pep_found = data.get("CC", 0)
        total_found = mhc_found + pep_found
        
        return mhc_found, pep_found, total_found
    except Exception as e:
        return 0, 0, 0

def extract_short_name(full_name):
    """
    Usa Regex para extrair o nome entre 'pmhc_' e o proximo '_'.
    Ex: pmhc_mage3_5brz... -> mage3
    Ex: pmhc_titin_5bs0... -> titin
    """
    # Padrão: comece com pmhc_, capture tudo que não for _ (group 1), termine com _
    match = re.search(r"pmhc_([^_]+)_", full_name)
    if match:
        return match.group(1)
    return full_name  # Retorna o original se o padrão não bater

def plot_heatmap(df_matrix, title, out_path):
    """
    Gera um heatmap estilo matriz de confusão.
    """
    plt.figure(figsize=(5, 5)) # Tamanho quadrado e compacto
    sns.set_context("notebook", font_scale=1.4) # Aumentei a fonte para ficar legível
    
    ax = sns.heatmap(
        df_matrix, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        cbar=False, 
        linewidths=1.5, 
        linecolor='black',
        square=True,
        annot_kws={"size": 16, "weight": "bold"} # Números grandes
    )
    
    plt.title(title, pad=25, fontsize=16, fontweight='bold')
    
    # Eixo X no topo
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    
    # Rotação 0 para ficar horizontal (já que o nome agora é curto)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gerado: {out_path.name}")

def process_csv(csv_path):
    csv_file = Path(csv_path)
    if not csv_file.exists():
        print("Arquivo não encontrado.")
        return

    df = pd.read_csv(csv_file)
    out_dir = csv_file.parent / "matrices_clean_output"
    out_dir.mkdir(exist_ok=True)

    # 1. Parsear Contagens
    parsed_data = df["unique_nodes_per_chain"].apply(parse_counts)
    df["mhc_found_parsed"] = [p[0] for p in parsed_data]
    df["pep_found_parsed"] = [p[1] for p in parsed_data]
    df["total_found_parsed"] = [p[2] for p in parsed_data]

    # 2. Limpar Nomes (APLICANDO O REGEX AQUI)
    # Criamos uma coluna nova 'short_name' para usar nos índices
    df["short_name"] = df["protein_name"].apply(extract_short_name)

    groups = df.groupby(["component_id", "frame_id"])

    for (comp_id, frame_id), group in groups:
        # Usamos o nome curto agora
        proteins = group["short_name"].unique()
        
        mat_total = pd.DataFrame(0, index=proteins, columns=proteins)
        mat_pep   = pd.DataFrame(0, index=proteins, columns=proteins)
        mat_mhc   = pd.DataFrame(0, index=proteins, columns=proteins)

        for protein in proteins:
            # Filtramos pelo nome curto
            row = group[group["short_name"] == protein].iloc[0]
            
            # Diagonal (Original)
            mat_total.loc[protein, protein] = row["total_nodes_original"]
            mat_pep.loc[protein, protein]   = row["pep_orig_nodes"]
            mat_mhc.loc[protein, protein]   = row["mhc_orig_nodes"]
            
            # Fora da Diagonal (Encontrado)
            others = [p for p in proteins if p != protein]
            for other in others:
                mat_total.loc[protein, other] = row["total_found_parsed"]
                mat_pep.loc[protein, other]   = row["pep_found_parsed"]
                mat_mhc.loc[protein, other]   = row["mhc_found_parsed"]

        base_name = f"comp{comp_id}_frame{frame_id}"

        plot_heatmap(mat_total, f"Nós Totais\n(Comp {comp_id} Frame {frame_id})", out_dir / f"{base_name}_TOTAL.png")
        plot_heatmap(mat_pep, f"Nós do Peptídeo\n(Comp {comp_id} Frame {frame_id})", out_dir / f"{base_name}_PEPTIDE.png")
        plot_heatmap(mat_mhc, f"Nós do MHC\n(Comp {comp_id} Frame {frame_id})", out_dir / f"{base_name}_MHC.png")


process_csv("/home/elementare/GithubProjects/pMHC_graph/data/magextitin/ALL_nodes_per_protein.csv")
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Gera matrizes 2x2 (Total, Pep, MHC) a partir do CSV.")
#     parser.add_argument("csv_path", type=str, help="Caminho do arquivo .csv")
#     args = parser.parse_args()

#     process_csv(args.csv_path)