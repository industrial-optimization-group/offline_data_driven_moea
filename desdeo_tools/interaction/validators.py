import pandas as pd


class ValidationError(Exception):
    """Raised when an error related to the validation is encountered.
    """


def validate_ref_point_with_ideal_and_nadir(
    dimensions_data: pd.DataFrame, reference_point: pd.DataFrame
):
    validate_ref_point_dimensions(dimensions_data, reference_point)
    validate_ref_point_data_type(reference_point)
    validate_ref_point_with_ideal(dimensions_data, reference_point)
    validate_with_ref_point_nadir(dimensions_data, reference_point)


def validate_ref_point_with_ideal(
    dimensions_data: pd.DataFrame, reference_point: pd.DataFrame
):
    validate_ref_point_dimensions(dimensions_data, reference_point)
    ideal_fitness = dimensions_data.loc["ideal"] * dimensions_data.loc["minimize"]
    ref_point_fitness = reference_point * dimensions_data.loc["minimize"]
    if not (ideal_fitness <= ref_point_fitness).all(axis=None):
        problematic_columns = ideal_fitness.index[
            (ideal_fitness > ref_point_fitness).values.tolist()[0]
        ].values
        msg = (
            f"Reference point should be worse than or equal to the ideal point\n"
            f"The following columns have problematic values: {problematic_columns}"
        )
        raise ValidationError(msg)


def validate_with_ref_point_nadir(
    dimensions_data: pd.DataFrame, reference_point: pd.DataFrame
):
    validate_ref_point_dimensions(dimensions_data, reference_point)
    nadir_fitness = dimensions_data.loc["nadir"] * dimensions_data.loc["minimize"]
    ref_point_fitness = reference_point * dimensions_data.loc["minimize"]
    if not (ref_point_fitness <= nadir_fitness).all(axis=None):
        problematic_columns = nadir_fitness.index[
            (nadir_fitness < ref_point_fitness).values.tolist()[0]
        ].values
        msg = (
            f"Reference point should be better than or equal to the nadir point\n"
            f"The following columns have problematic values: {problematic_columns}"
        )
        raise ValidationError(msg)


def validate_ref_point_dimensions(
    dimensions_data: pd.DataFrame, reference_point: pd.DataFrame
):
    if not dimensions_data.shape[1] == reference_point.shape[1]:
        msg = (
            f"There is a mismatch in the number of columns of the dataframes.\n"
            f"Columns in dimensions data: {dimensions_data.columns}\n"
            f"Columns in the reference point provided: {reference_point.columns}"
        )
        raise ValidationError(msg)
    if not all(dimensions_data.columns == reference_point.columns):
        msg = (
            f"There is a mismatch in the column names of the dataframes.\n"
            f"Columns in dimensions data: {dimensions_data.columns}\n"
            f"Columns in the reference point provided: {reference_point.columns}"
        )
        raise ValidationError(msg)


def validate_ref_point_data_type(reference_point: pd.DataFrame):
    for dtype in reference_point.dtypes:
        if not ((dtype == int) or (dtype == float)):
            msg = (
                f"Type of data in reference point dataframe should be int or float.\n"
                f"Provided datatype: {dtype}"
            )
            raise ValidationError(msg)
