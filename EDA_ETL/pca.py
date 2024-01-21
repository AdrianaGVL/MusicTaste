import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Load dataset
data = pd.read_csv('Features/new_data/df_enorm.csv', sep=';', decimal=",", index_col=None)

Application = 'R'   # R for Regression | C for Classification
n_components = 16
num_features_classif = 1 + n_components     # Total classification features = 5
num_features_regress = 7 + n_components     # Total regression features = 28

X = ''
features = ''
if Application == 'R':
    X = data.iloc[:, 7:num_features_regress]
    features = X.columns
else:
    X = data.iloc[:, 1:num_features_classif]
    features = X.columns

# # Elbow rule
# pca = PCA()
# pca.fit(X)
#
# ############### JUST TO STUDY THE NUMBER OF COMPONENTS ##############
# # Accumulative variance
# cumulative_explained_variance = pca.explained_variance_ratio_.cumsum()
#
# # Elbow rule chart
# plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--')
# plt.axhline(y=0.80, color="r", linestyle="-")
# plt.axhline(y=0.90, color="purple", linestyle="-")
# plt.axhline(y=0.95, color="orange", linestyle="-")
# plt.xlabel('Number of principal components')
# plt.ylabel('Explained Accumulative Variance')
# plt.title('Principal Components Analysis (PCA) - Elbow rule')
# plt.savefig(f"Features/new_data/PCA/{Application}/Number_of_components.png")
# plt.show()
# ############### JUST TO STUDY THE NUMBER OF COMPONENTS ##############

# PCA model
components=[]
n=1
while n<=n_components:
    components.append("PC"+str(n))
    n=n+1

pca = PCA(n_components=n_components)
pc = pca.fit_transform(X)

# Biplot
plt.scatter(pc[:, 0], pc[:, 1], alpha=0.7)
# Arrow adding
for i, variable in enumerate(pca.components_.T):
    plt.arrow(0, 0, variable[0], variable[1], color='yellow', alpha=0.5)
    plt.text(variable[0] * 1.3, variable[1] * 1.3, f'{features[i]}', color='black', ha='center', va='center', fontsize=3.5)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.savefig(f"Features/new_data/PCA/{Application}/Biplot_{n_components}Components.png")
plt.show()

# Dataframe with PCA results and latex code
pca_df = pd.DataFrame(data=pc, columns=components)
print(pca_df.head())
# latex_code = tabulate(pca_df, headers='keys', tablefmt='latex_raw')
# if os.path.isfile(f'Features/new_data/PCA/pca_results.txt'):
#     with open(f'Features/new_data/PCA/pca_results.txt', 'a') as file:
#         file.write(f'\nPCA Results file:\n')
#         file.write(latex_code)
# else:
#     with open(f'Features/new_data/PCA/pca_results.txt', 'w') as file:
#         file.write(f'PCA Results file:\n')
#         file.write(latex_code)

# Explore variance per component
explained_variance = pca.explained_variance_ratio_
print(f'Varianza explicada por cada componente: {explained_variance}')

components_df = pd.DataFrame(pca.components_, columns=features, index=components)
print(components_df)

# Explicative variance chart per component
plt.bar(features, explained_variance)
plt.xlabel("Principal Components")
plt.ylabel("Explained Variance")
plt.title("Explained Variance per Principal Components")
plt.xticks(fontsize=4.5)
plt.savefig(f"Features/new_data/PCA/{Application}/ExplainedComponents_{n_components}Components.png")
plt.show()

fig, ax = plt.subplots()
info = 'Biplot' # Arrows or Biplot
def update(frame):
    if info == 'Arrows':
        ax.clear()
    else:
        ax.clear()
        plt.scatter(pc[:, 0], pc[:, 1], alpha=0.7)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(features[frame])

    # Dibujar una sola flecha en cada frame
    variable = pca.components_.T[frame]
    ax.arrow(0, 0, variable[0], variable[1], color='yellow', alpha=0.5)
    ax.text(variable[0] * 1.3, variable[1] * 1.3, f'{features[frame]}', color='black', ha='center', va='center', fontsize=8)

# Crear la animación
frames = pca.components_.shape[1]
animation = FuncAnimation(fig, update, frames=frames, interval=1000, repeat=False)
animation.save(f'Features/new_data/PCA/{Application}/{info}_animation.gif', writer='pillow')