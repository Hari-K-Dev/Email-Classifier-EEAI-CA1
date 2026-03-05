# This file contains some variable names you need to use in overall project.
#For example, this will contain the name of dataframe columns we will working on each file
class Config:
    # Input Columns
    TICKET_SUMMARY = 'Ticket Summary'
    INTERACTION_CONTENT = 'Interaction content'

    # CSV column name mapping to internal names
    TYPE_COL_MAP = {
        'y1': 'Type 1',
        'y2': 'Type 2',
        'y3': 'Type 3',
        'y4': 'Type 4',
    }

    # Type Columns to test
    TYPE_COLS = ['y2', 'y3', 'y4']
    CLASS_COL = 'y2'
    GROUPED = 'y1'

    # Chained target definitions for Design Choice 1
    CHAINED_LEVELS = [
        ['y2'],              # Level 1: Type 2 alone
        ['y2', 'y3'],        # Level 2: Type 2 + Type 3
        ['y2', 'y3', 'y4'],  # Level 3: Type 2 + Type 3 + Type 4
    ]

    # Separator for chained labels
    CHAIN_SEP = ' + '

    # Rare class threshold (remove classes with <= this many instances)
    RARE_THRESHOLD = 1

    # Random seed
    SEED = 0

    # Data file paths
    DATA_FILES = ['data/AppGallery.csv', 'data/Purchasing.csv']