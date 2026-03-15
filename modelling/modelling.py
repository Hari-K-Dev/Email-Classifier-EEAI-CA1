from model.randomforest import RandomForest
from modelling.data_model import Data
from Config import Config
from utils import create_chained_label


def model_predict(data, df, name):
    """Chained Multi-Output Classification (Design Choice 1).
    For each level in Config.CHAINED_LEVELS:
      1. Create a chained target column in df
      2. Build a Data object with that target
      3. Instantiate a RandomForest model
      4. Train, predict, and evaluate
    """
    # Get the full embeddings matrix from the initial data object
    X = data.get_embeddings()

    for i, level_cols in enumerate(Config.CHAINED_LEVELS):
        level_name = Config.CHAIN_SEP.join(level_cols)
        print(f"\n{'='*60}")
        print(f"Level {i+1}: Classifying {level_name}")
        print(f"{'='*60}")

        # Create chained label column
        chained_col_name = '_'.join(level_cols)
        df[chained_col_name] = create_chained_label(df, level_cols, Config.CHAIN_SEP)

        # Create Data object for this level
        level_data = Data(X, df, target_col=chained_col_name)

        # Instantiate, train, predict
        model_name = f"RandomForest_Level{i+1}_{chained_col_name}"
        rf = RandomForest(
            model_name=model_name,
            embeddings=level_data.get_embeddings(),
            y=level_data.get_type()
        )
        rf.train(level_data)
        rf.predict(level_data.get_X_test())

        # Evaluate
        model_evaluate(rf, level_data)


def model_evaluate(model, data):
    model.print_results(data)
