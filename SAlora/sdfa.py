import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Plot styling (use common sans-serif and LaTeX-style math text)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'cm'

def draw_robust_figure():
    # Create a fixed-aspect canvas to produce a stable layout
    fig = plt.figure(figsize=(14, 8), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 10)
    ax.axis('off') # turn off axes

    # color definitions
    c_blue = '#1f77b4'
    c_red = '#d62728'   # Deep / High rank
    c_green = '#2ca02c' # Shallow / Low rank
    c_gray = '#eeeeee'
    c_border = '#333333'

    # ==========================================
    # Section 1: Heterogeneity Analysis (left)
    # ==========================================
    ax.text(3, 9.2, "1. Heterogeneity Analysis", fontsize=14, fontweight='bold', color='#444', ha='center')

    # Deep Layer Box
    rect_deep = patches.FancyBboxPatch((1, 6.5), 4, 1.5, boxstyle="round,pad=0.1", fc=c_gray, ec=c_border)
    ax.add_patch(rect_deep)
    ax.text(3, 7.4, "Layer $L$ (Deep)", fontsize=12, fontweight='bold', ha='center')
    ax.text(3, 6.9, "Semantic Features", fontsize=10, fontstyle='italic', ha='center')

    # Shallow Layer Box
    rect_shallow = patches.FancyBboxPatch((1, 2.0), 4, 1.5, boxstyle="round,pad=0.1", fc=c_gray, ec=c_border)
    ax.add_patch(rect_shallow)
    ax.text(3, 2.9, "Layer $1$ (Shallow)", fontsize=12, fontweight='bold', ha='center')
    ax.text(3, 2.4, "Surface Features", fontsize=10, fontstyle='italic', ha='center')

    # Ellipsis
    ax.text(3, 5.0, r"$\vdots$", fontsize=24, ha='center', color='#888')

    # SVD sketch (manual curves to avoid subplots)
    # Deep SVD (flat decay)
    ax.text(5.5, 7.8, "SVD", fontsize=9, color=c_blue)
    ax.annotate("", xy=(6.5, 7.25), xytext=(5.1, 7.25), arrowprops=dict(arrowstyle="->", color=c_border))
    
    # Axes
    ax.plot([6.5, 8.5], [6.5, 6.5], 'k-', lw=1) # x
    ax.plot([6.5, 6.5], [6.5, 8.0], 'k-', lw=1) # y
    # Curve
    x_svd = np.linspace(6.6, 8.4, 20)
    y_svd_deep = 7.8 - 0.2 * (x_svd - 6.5)  # gentle decay
    ax.plot(x_svd, y_svd_deep, color=c_red, lw=2)
    ax.text(7.5, 8.2, r"High $s_L$", color=c_red, fontsize=9)

    # Shallow SVD (Steep curve)
    ax.text(5.5, 3.3, "SVD", fontsize=9, color=c_blue)
    ax.annotate("", xy=(6.5, 2.75), xytext=(5.1, 2.75), arrowprops=dict(arrowstyle="->", color=c_border))
    
    # Axes
    ax.plot([6.5, 8.5], [2.0, 2.0], 'k-', lw=1) # x
    ax.plot([6.5, 6.5], [2.0, 3.5], 'k-', lw=1) # y
    # Curve
    y_svd_shallow = 2.0 + 1.4 / (x_svd - 6.2)  # steep decay
    y_svd_shallow = np.clip(y_svd_shallow, 2.0, 3.4)
    ax.plot(x_svd, y_svd_shallow, color=c_green, lw=2)
    ax.text(7.5, 3.7, r"Low $s_1$", color=c_green, fontsize=9)


    # ==========================================
    # Section 2: Principled Mapping (center)
    # ==========================================
    ax.text(10, 9.2, "2. Principled Mapping", fontsize=14, fontweight='bold', color=c_blue, ha='center')

    # Central processing box
    box_x, box_y, box_w, box_h = 8.5, 3.0, 4.5, 4.5
    rect_engine = patches.FancyBboxPatch((box_x, box_y), box_w, box_h, boxstyle="round,pad=0.2", fc='#f0f8ff', ec=c_blue, lw=2, linestyle='--')
    ax.add_patch(rect_engine)
    
    ax.text(box_x + box_w/2, box_y + 4.0, "SA-LoRA Allocation", fontsize=12, fontweight='bold', ha='center', color=c_blue)

    # Formulae
    ax.text(box_x + box_w/2, box_y + 3.0, r"1. Metric:", fontsize=10, ha='center', color='#555')
    ax.text(box_x + box_w/2, box_y + 2.4, r"$s_l = \frac{\|W^*\|_F^2}{\|W^*\|_2^2}$", fontsize=13, ha='center')

    ax.text(box_x + box_w/2, box_y + 1.5, r"2. Mapping:", fontsize=10, ha='center', color='#555')
    ax.text(box_x + box_w/2, box_y + 0.9, r"$\mu_l \propto (s_l)^p$", fontsize=13, ha='center')
    
    # Budget constraint annotation
    rect_budget = patches.FancyBboxPatch((box_x + 0.5, 2.5), 3.5, 0.8, boxstyle="round,pad=0.1", fc='white', ec=c_red, lw=1.5)
    ax.add_patch(rect_budget)
    ax.text(box_x + box_w/2, 3.0, "Budget Constraint", fontsize=9, fontweight='bold', color=c_red, ha='center')
    ax.text(box_x + box_w/2, 2.6, r"$\frac{1}{L}\sum \mu_l = 1$", fontsize=11, ha='center')


    # ==========================================
    # Section 3: Adaptive Injection (right)
    # ==========================================
    ax.text(16.5, 9.2, "3. Adaptive Injection", fontsize=14, fontweight='bold', color='#444', ha='center')

    # Deep LoRA (High Multiplier)
    # Frozen path
    ax.text(15, 8.0, r"Frozen $W^*$", ha='center', fontsize=10, color='#888')
    ax.annotate("", xy=(18.5, 7.25), xytext=(13.5, 7.25), arrowprops=dict(arrowstyle="->", color=c_border, lw=1.5))
    
    # LoRA Path
    rect_A = patches.Rectangle((14.5, 6.0), 0.8, 1.2, fc='white', ec=c_blue)
    rect_B = patches.Rectangle((16.0, 6.0), 0.8, 1.2, fc='white', ec=c_blue)
    ax.add_patch(rect_A)
    ax.add_patch(rect_B)
    ax.text(14.9, 6.6, "A", ha='center', va='center', fontweight='bold')
    ax.text(16.4, 6.6, "B", ha='center', va='center', fontweight='bold')
    
    # Arrows through LoRA
    ax.annotate("", xy=(14.5, 6.6), xytext=(13.5, 6.6), arrowprops=dict(arrowstyle="->", color=c_border))
    ax.annotate("", xy=(16.0, 6.6), xytext=(15.3, 6.6), arrowprops=dict(arrowstyle="->", color=c_border))
    ax.annotate("", xy=(18.0, 6.6), xytext=(16.8, 6.6), arrowprops=dict(arrowstyle="->", color=c_border))
    
    # Summation
    circle_plus = patches.Circle((18.5, 7.25), 0.25, fc='white', ec='black')
    ax.add_patch(circle_plus)
    ax.text(18.5, 7.25, "+", ha='center', va='center', fontsize=14)
    ax.annotate("", xy=(18.5, 7.0), xytext=(18.5, 6.6), arrowprops=dict(arrowstyle="->", color=c_border))
    
    # Multiplier Injection (RED - Large)
    circle_mu = patches.Circle((15.6, 5.5), 0.5, fc='#ffeeee', ec=c_red, lw=2)
    ax.add_patch(circle_mu)
    ax.text(15.6, 5.5, r"$\times \mu_L$", color=c_red, fontweight='bold', ha='center', va='center')
    ax.text(15.6, 4.8, r"$(\mu_L > 1)$", color=c_red, fontsize=9, ha='center')
    
    # Connection lines from engine to multiplier
    ax.annotate("", xy=(15.6, 6.0), xytext=(13.0, 5.25), 
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", color=c_blue, lw=2, ls='--'))


    # Shallow LoRA (Low Multiplier)
    # Frozen path
    ax.annotate("", xy=(18.5, 2.75), xytext=(13.5, 2.75), arrowprops=dict(arrowstyle="->", color=c_border, lw=1.5))
    
    # LoRA Path
    rect_As = patches.Rectangle((14.5, 1.5), 0.8, 1.2, fc='white', ec=c_blue)
    rect_Bs = patches.Rectangle((16.0, 1.5), 0.8, 1.2, fc='white', ec=c_blue)
    ax.add_patch(rect_As)
    ax.add_patch(rect_Bs)
    ax.text(14.9, 2.1, "A", ha='center', va='center', fontweight='bold')
    ax.text(16.4, 2.1, "B", ha='center', va='center', fontweight='bold')
    
    # Arrows
    ax.annotate("", xy=(14.5, 2.1), xytext=(13.5, 2.1), arrowprops=dict(arrowstyle="->", color=c_border))
    ax.annotate("", xy=(16.0, 2.1), xytext=(15.3, 2.1), arrowprops=dict(arrowstyle="->", color=c_border))
    ax.annotate("", xy=(18.0, 2.1), xytext=(16.8, 2.1), arrowprops=dict(arrowstyle="->", color=c_border))

    # Summation
    circle_pluss = patches.Circle((18.5, 2.75), 0.25, fc='white', ec='black')
    ax.add_patch(circle_pluss)
    ax.text(18.5, 2.75, "+", ha='center', va='center', fontsize=14)
    ax.annotate("", xy=(18.5, 2.5), xytext=(18.5, 2.1), arrowprops=dict(arrowstyle="->", color=c_border))

    # Multiplier Injection (GREEN - Small)
    circle_mus = patches.Circle((15.6, 1.0), 0.5, fc='#eeffee', ec=c_green, lw=2)
    ax.add_patch(circle_mus)
    ax.text(15.6, 1.0, r"$\times \mu_1$", color=c_green, fontweight='bold', ha='center', va='center')
    ax.text(15.6, 0.3, r"$(\mu_1 < 1)$", color=c_green, fontsize=9, ha='center')

    # Connection lines from engine to multiplier
    ax.annotate("", xy=(15.1, 1.0), xytext=(13.0, 4.0), 
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color=c_blue, lw=2, ls='--'))

    # Final Save
    plt.tight_layout()
    plt.savefig("sa_lora_fixed.pdf", bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    draw_robust_figure()