from data.data_source import DataSource


def main():
    print(f"Starting...")
    src = DataSource()
    df = src.get_data(num_elements=1000000)
    print(f"Got Pandas dataframe with {df.size} elements, top 10:")
    print(df.head(10))


main()