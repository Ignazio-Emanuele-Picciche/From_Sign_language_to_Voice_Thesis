import pandas as pd


def convert_xlsx_to_tsv(input_xlsx_path, output_tsv_path):
    """
    Converts an Excel file to a TSV file with specific columns.

    Parameters:
        input_xlsx_path (str): Path to the input Excel file.
        output_tsv_path (str): Path to the output TSV file.
    """
    try:
        # Read the Excel file
        df = pd.read_excel(input_xlsx_path)

        # Rename columns to match the desired format
        df = df.rename(
            columns={"VIDEO_NAME": "id", "START": "duration", "SENTENCE": "text"}
        )

        # Select only the required columns
        df = df[["id", "duration", "text"]]

        # Save the DataFrame to a TSV file
        df.to_csv(output_tsv_path, sep="\t", index=False)

        print(f"Successfully converted {input_xlsx_path} to {output_tsv_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # Example usage
    input_xlsx_path = "data/raw/val/how2sign_val.xlsx"  # Replace with the actual path to your Excel file
    output_tsv_path = "colab_upload/How2Sign_SONAR/manifests/val.tsv"  # Replace with the desired path for the TSV file

    convert_xlsx_to_tsv(input_xlsx_path, output_tsv_path)
