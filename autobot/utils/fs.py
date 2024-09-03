def read_file(filename: str):
    try:
        with open(filename, "r") as f:
            return f.read()
    except IOError:
        raise FileNotFoundError(f"Read {filename} failed.")
