
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib import rcParams

# Publication settings
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12
rcParams['axes.labelsize'] = 13
rcParams['axes.titlesize'] = 14
rcParams['savefig.dpi'] = 600

sns.set_style("whitegrid")


def load_features(dataset_name, base_dir):
    """Load pre-computed features."""
    csv_path = Path(base_dir) / f"{dataset_name}_features_ALL.csv"
    if csv_path.exists():
        print(f"[+] Loading {dataset_name} from {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Features:
        # n1_h = OUT-degree (HEAD)
        # R1_h = Relation diversity (HEAD)
        # b1 = Breadth edges (UNDIRECTED) - already in CSV
        # d1 = Depth edges (UNDIRECTED) - already in CSV
        
        return df
    else:
        print(f"  ⚠️  Not found: {csv_path}")
        return None


def create_combined_boxplots(all_data, outdir):
    """Create combined boxplots (4 panels) - Labels INSIDE the plots."""
    print("\n[+] Creating combined boxplots (LaTeX-ready)...")
    
    features = [
        ('n1_h', 'OUT-degree', True),           # n¹ₕ (HEAD)
        ('R1_h', 'Relation Diversity', False),  # R¹ₕ (HEAD)
        ('b1', 'Breadth Edges', True),          # b¹ (UNDIRECTED)
        ('d1', 'Depth Edges', True)             # d¹ (UNDIRECTED)
    ]
    
    colors = {
        'FB15': '#FFB347',
        'NELL': '#4A90E2',
        'HealthKG': '#27AE60'
    }
    
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    
    for idx, (feat_col, feat_label, use_log) in enumerate(features):
        ax = axes[idx]
        
        plot_data = []
        labels = []
        color_list = []
        
        for dataset in ['FB15', 'NELL', 'HealthKG']:
            if dataset in all_data:
                df = all_data[dataset]
                values = df[feat_col].values
                
                # For log scale, filter zeros
                if use_log:
                    values = values[values > 0]
                
                plot_data.append(values)
                labels.append(dataset)
                color_list.append(colors[dataset])
        
        if not plot_data:
            continue
        
        # Create box plot
        bp = ax.boxplot(
            plot_data, labels=labels, patch_artist=True, showfliers=False,
            widths=0.6, medianprops=dict(color='black', linewidth=2.5),
            boxprops=dict(linewidth=2.0), whiskerprops=dict(linewidth=2.0),
            capprops=dict(linewidth=2.0)
        )
        
        for patch, color in zip(bp['boxes'], color_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(2.0)
        
        # NO title - add label INSIDE the plot instead
        ax.set_ylabel('Value', fontsize=13, fontweight='bold')
        
        # Add feature label INSIDE the plot (top-left corner)
        ax.text(0.05, 0.95, f'({chr(97+idx)}) {feat_label}', 
                transform=ax.transAxes,
                fontsize=13, fontweight='bold',
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', 
                         facecolor='white', 
                         edgecolor='black',
                         linewidth=1.5,
                         alpha=0.9))
        
        if use_log:
            ax.set_yscale('log')
        
        ax.grid(True, alpha=0.25, axis='y', linewidth=1.0, linestyle='--')
        ax.tick_params(axis='both', which='major', labelsize=11, width=1.5, length=6)
        
        for spine in ax.spines.values():
            spine.set_linewidth(1.8)
            spine.set_color('black')
    
    plt.tight_layout()
    
    # Save
    output_png = outdir / "boxplots_mixed_4panels.png"
    output_eps = outdir / "boxplots_mixed_4panels.eps"
    
    plt.savefig(output_png, dpi=600, bbox_inches='tight', facecolor='white')
    plt.savefig(output_eps, format='eps', bbox_inches='tight')
    
    print(f"  ✅ PNG: {output_png}")
    print(f"  ✅ EPS: {output_eps}")
    
    plt.close()


def create_combined_histograms(all_data, outdir):
    """Create histograms (3 panels per feature) - MIXED features."""
    print("\n[+] Creating combined histograms (LaTeX-ready)...")
    
    features = ['n1_h', 'R1_h', 'b1', 'd1']
    feature_labels = {
        'n1_h': 'n¹ₕ (OUT-degree)',
        'R1_h': 'R¹ₕ (Relation Diversity)',
        'b1': 'b¹ (Breadth Edges)',
        'd1': 'd¹ (Depth Edges)'
    }
    
    colors = {
        'FB15': '#FFB347',
        'NELL': '#4A90E2',
        'HealthKG': '#27AE60'
    }
    
    for feat in features:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, dataset in enumerate(['FB15', 'NELL', 'HealthKG']):
            if dataset not in all_data:
                continue
            
            ax = axes[idx]
            df = all_data[dataset]
            values = df[feat].values
            
            # Remove zeros
            non_zero = values[values > 0]
            
            if len(non_zero) > 0:
                ax.hist(non_zero, bins=50, color=colors[dataset], 
                       alpha=0.7, edgecolor='black', linewidth=1.2)
                ax.set_xlabel(feature_labels[feat], fontsize=12, fontweight='bold')
                ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
                ax.set_title(f'{dataset}\n(n={len(non_zero):,} non-zero)', 
                           fontsize=13, fontweight='bold')
                ax.set_yscale('log')
                
                # Add statistics
                textstr = f'Mean: {non_zero.mean():.2f}\nMedian: {np.median(non_zero):.2f}'
                ax.text(0.65, 0.95, textstr, transform=ax.transAxes, 
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            else:
                ax.text(0.5, 0.5, 'All zeros', ha='center', va='center',
                       fontsize=14, fontweight='bold')
                ax.set_title(f'{dataset}\n(All zeros)', fontsize=13, fontweight='bold')
            
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        safe_name = feat.replace('_', '-')
        output_png = outdir / f"histogram_mixed_{safe_name}_3panels.png"
        output_eps = outdir / f"histogram_mixed_{safe_name}_3panels.eps"
        
        plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(output_eps, format='eps', bbox_inches='tight')
        
        print(f"  ✅ {feat}: PNG + EPS")
        
        plt.close()


def create_latex_template(outdir):
    """Generate LaTeX template."""
    print("\n[+] Creating LaTeX template...")
    
    template = r"""
% LaTeX Template for φ-feature Figures (MIXED version)
% Features: n¹ₕ (HEAD), R¹ₕ (HEAD), b¹ (undirected), d¹ (undirected)

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{boxplots_mixed_4panels.eps}
\caption{Distribution of $\phi$-features across datasets. 
(a) OUT-degree $n^1_h$ (directed), (b) Relation diversity $R^1_h$ (directed), 
(c) Breadth edges $b^1$ (undirected), (d) Depth edges $d^1$ (undirected).}
\label{fig:phi-mixed}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{histogram_mixed_n1-h_3panels.eps}
\caption{Distribution of OUT-degree $n^1_h$ across datasets.}
\label{fig:hist-n1h}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{histogram_mixed_R1-h_3panels.eps}
\caption{Distribution of Relation diversity $R^1_h$ across datasets.}
\label{fig:hist-r1h}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{histogram_mixed_b1_3panels.eps}
\caption{Distribution of Breadth edges $b^1$ across datasets.}
\label{fig:hist-b1}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{histogram_mixed_d1_3panels.eps}
\caption{Distribution of Depth edges $d^1$ across datasets.}
\label{fig:hist-d1}
\end{figure}
"""
    
    latex_path = outdir / "latex_template_mixed.tex"
    with open(latex_path, 'w') as f:
        f.write(template)
    
    print(f"  ✅ LaTeX template: {latex_path}")


def main():
    base_dir = Path("/home/harrouch/Phi_Distributions_Incremental")
    
    # Load all datasets
    all_data = {}
    for dataset_name in ['FB15', 'NELL', 'HealthKG']:
        df = load_features(dataset_name, base_dir)
        if df is not None:
            all_data[dataset_name] = df
    
    if not all_data:
        print("❌ No data found! Run phi_ultra_fast_DIAGNOSTIC.py first.")
        return
    
    print(f"\n[+] Loaded {len(all_data)} datasets: {list(all_data.keys())}")
    
    # Create visualizations
    create_combined_boxplots(all_data, base_dir)
    create_combined_histograms(all_data, base_dir)
    create_latex_template(base_dir)
    
    print("\n" + "="*80)
    print("✅ MIXED VISUALIZATIONS CREATED!")
    print("="*80)
    print("\nGenerated files (PNG + EPS):")
    print("  BOXPLOTS:")
    print("    - boxplots_mixed_4panels.eps")
    print("  HISTOGRAMS:")
    print("    - histogram_mixed_n1-h_3panels.eps  (n¹ₕ - directed HEAD)")
    print("    - histogram_mixed_R1-h_3panels.eps  (R¹ₕ - directed HEAD)")
    print("    - histogram_mixed_b1_3panels.eps    (b¹ - undirected)")
    print("    - histogram_mixed_d1_3panels.eps    (d¹ - undirected)")
    print("  TEMPLATE:")
    print("    - latex_template_mixed.tex")
    print("\nFeature configuration:")
    print("  - n¹ₕ and R¹ₕ: DIRECTED (HEAD only)")
    print("  - b¹ and d¹: UNDIRECTED (all neighbors)")
    print("  - Labels are INSIDE the boxplots")


if __name__ == "__main__":
    main()