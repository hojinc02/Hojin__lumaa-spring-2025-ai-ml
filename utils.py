import argparse

def get_args(): 
    """Parse arguments. """
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--desc", help="Input description", 
                        type=str, required=True)
    parser.add_argument("-c", "--count", help="Number of movies to recommend",
                        default=5, type=int)

    args = parser.parse_args()

    return args
