import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from Data_Preprocessing import read_data, preprocess


# Path to results/figures
FIGURES_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


train_df = read_data()
def eda(df):
    print("Shape of data:", df.shape)
    print("\nData types:\n", df.dtypes)
    print(df.describe())
    print("\nMissing Values:\n", df.isnull().sum())

    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')

    plt.title("Visualizing Missing Data", fontsize=14)
    plt.xlabel("Columns", fontsize=12)
    plt.ylabel("Rows", fontsize=12)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "missing_values.png", dpi=300, bbox_inches="tight")

    plt.show()
    plt.close()

    # Age Distribution
    plt.figure(figsize=(15, 5))
    sns.histplot(df['Age'].dropna(), kde=True)
    plt.title('Age Distribution')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "age_distribution.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    # Create the figure and the two axes (sub-containers)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot Age on the first axis (axes[0])
    sns.boxplot(x=df['Age'], ax=axes[0], color='skyblue')
    axes[0].set_title('Age Outliers Detection')

    # Plot Fare on the second axis (axes[1])
    sns.boxplot(x=df['Fare'], ax=axes[1], color='salmon')
    axes[1].set_title('Fare Outliers Detection')
    plt.show()

    # 3. Correlation Heatmap (numeric only)
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Correlation Heatmap")
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png",dpi=300,bbox_inches="tight")

    plt.show()
    plt.close()

    # Categorical Analysis
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.countplot(x='Sex', hue='Survived', data=df, ax=axes[0])
    axes[0].set_title('Survival by Sex')

    sns.countplot(x='Pclass', hue='Survived', data=df, ax=axes[1])
    axes[1].set_title('Survival by Pclass')

    sns.countplot(x='Embarked', hue='Survived', data=df, ax=axes[2])
    axes[2].set_title('Survival by Embarked')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "survival_analysis.png",dpi=300,bbox_inches="tight")

    plt.show()
    plt.close()

    sns.countplot(x='Survived', data=df)
    plt.show()
    plt.close()


if __name__ == "__main__":
    train_df = read_data()
    eda(train_df)
