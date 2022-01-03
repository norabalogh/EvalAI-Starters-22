import random
import sklearn.metrics as sm
import pandas as pd


def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    print("Starting Evaluation.....")
    """
    Evaluates the submission for a particular challenge phase and returns score
    Arguments:

        `test_annotations_file`: Path to test_annotation_file on the server
        `user_submission_file`: Path to file submitted by the user
        `phase_codename`: Phase to which submission is made

        `**kwargs`: keyword arguments that contains additional submission
        metadata that challenge hosts can use to send slack notification.
        You can access the submission metadata
        with kwargs['submission_metadata']

        Example: A sample submission metadata can be accessed like this:
        >>> print(kwargs['submission_metadata'])
        {
            'status': u'running',
            'when_made_public': None,
            'participant_team': 5,
            'input_file': 'https://abc.xyz/path/to/submission/file.json',
            'execution_time': u'123',
            'publication_url': u'ABC',
            'challenge_phase': 1,
            'created_by': u'ABC',
            'stdout_file': 'https://abc.xyz/path/to/stdout/file.json',
            'method_name': u'Test',
            'stderr_file': 'https://abc.xyz/path/to/stderr/file.json',
            'participant_team_name': u'Test Team',
            'project_url': u'http://foo.bar',
            'method_description': u'ABC',
            'is_public': False,
            'submission_result_file': 'https://abc.xyz/path/result/file.json',
            'id': 123,
            'submitted_at': u'2017-03-20T19:22:03.880652Z'
        }
    """
    solution_df = pd.read_csv(test_annotation_file)
    submission_df = pd.read_csv(user_submission_file)
    print("Submission shape: ", submission_df.shape)
    if submission_df.shape[0] != solution_df.shape[0]:
        raise Exception("Sumbmission number of rows are incorrect.")

    if submission_df.shape[1] !=2:
        raise Exception("Submission should have 2 columns.")
    submission_df.columns = ["ID", 'predicted_fuel_consumption_sum']

    solution_df = pd.merge(solution_df, submission_df,  on='ID', how='left')

    output = {}
    if phase_codename == "public":
        print("Evaluating for Public Phase")
        public_df = solution_df.loc[solution_df.Usage=="Public", :]
        mae_public = sm.mean_absolute_error(public_df["fuel_consumption_sum"], public_df["predicted_fuel_consumption_sum"])
        rmse_public = sm.mean_squared_error(public_df["fuel_consumption_sum"], public_df["predicted_fuel_consumption_sum"])

        print("Evaluating for Private Phase")
        private_df = solution_df.loc[solution_df.Usage=="Private", :]
        mae_private = sm.mean_absolute_error(private_df["fuel_consumption_sum"], private_df["predicted_fuel_consumption_sum"])
        rmse_private = sm.mean_squared_error(private_df["fuel_consumption_sum"], private_df["predicted_fuel_consumption_sum"])
        output["result"] = [
            {
                "public_split": {
                    "MAE": mae_public,
                    "RMSE": rmse_public
                }
            },
            {
                "private_split": {
                    "MAE": mae_private,
                    "RMSE": rmse_private
                }
            }
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]["public_split"]
        print("Completed evaluation.")
    return output

# if __name__ == "__main__":
#     res1 = evaluate("annotations/private_solution.csv", "challenge_data/public_sample_submission.csv", "public")
#     print("Result\n", res1)
#     res2 = evaluate("annotations/private_solution.csv", "challenge_data/public_sample_submission.csv", "private")
#     print("Result\n", res2)
