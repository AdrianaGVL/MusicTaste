import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
data = pd.read_csv('Features/new_data/df_norm.csv', sep=';', decimal=",", index_col=None)

Application = 'R'   # R for Regression | C for Classification

X = ''
features = ''
if Application == 'R':
    X = data.iloc[:, 6:23]
    features = X.columns
else:
    X = data.iloc[:, 1:6]
    features = X.columns

# Train the model
n_clusters = 5
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
fitted_kmeans = kmeans.fit_predict(X)

Centers = kmeans.cluster_centers_
df_centers = pd.DataFrame(data = Centers,
                  columns = features)
print(df_centers)

# Plot clusters
plt.subplots()
colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'orange', 'yellow', 'purple', 'brown', 'pink', 'black', 'indigo', 'olive', 'lightcoral', 'darkblue', 'lightgray']
for i in range(n_clusters):
    for j in range(len(features)-1):
        plt.scatter(X.iloc[fitted_kmeans == 0, j], X.iloc[fitted_kmeans == 0, j+1], s = 30, c = colors[j], label = f'Cluster {j}')
    plt.title(f'{features[i]}- {features[i + 1]}')
    plt.xlabel(f'{features[i]}')
    plt.ylabel(f'{features[i+1]}')
    plt.legend()
    plt.savefig(f"Features/new_data/Clustering/{Application}/Cluster_{features[i]}.png")
    plt.show()