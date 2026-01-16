import evaluate
import argparse
import json
import sys

evaluations = [('span_identifier', 'strict'), ('span', 'strict'), ('span', 'approx'), ('identifier', 'strict')]

if __name__ == "__main__":
    
    print(sys.argv)
    
    parser = argparse.ArgumentParser(description="Scoring script for NLM CellLink-NER task")
    parser.add_argument("--reference_path", "-r", type=str, required=True, help="path to directory or file containing the reference annotations, i.e. the annotations considered correct")
    parser.add_argument("--prediction_path", "-p", type=str, required=True, help="path to directory or file containing the predicted annotations, i.e. the annotations being evaluated")
    parser.add_argument("--output_json_filepath", type=str, required=True, help="filepath where output json file will be saved")
    args = parser.parse_args()
    
    results = dict()
    
    print("about to run evaluate.py")
    for evaluation_type, evaluation_method in evaluations:
        results[f"{evaluation_type},{evaluation_method},F1"] = \
            evaluate.main(reference_path=args.reference_path, prediction_path=args.prediction_path, 
                          evaluation_type=evaluation_type, evaluation_method=evaluation_method, input_annotation_type="none", 
                          logging_level="critical", verify_documents=True, skip_extra_pred_passages=True).f_score
        print(f"ran {evaluation_type},{evaluation_method}")
    
    print(f"finished running {len(evaluations)} evaluations. about to write output json.")
    with open(args.output_json_filepath, 'w') as writefp:
        json.dump(results, writefp)
    print("done")