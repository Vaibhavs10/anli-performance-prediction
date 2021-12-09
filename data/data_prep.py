import csv
import json

def process_jsonl(file_path):
    story_id = []
    obs1 = []
    obs2 = []
    hyp1 = []
    hyp2 = []

    with open(file_path) as json_file:
        for line in json_file:
            dict_line = json.loads(line)
            story_id.append(dict_line["story_id"])
            obs1.append(dict_line["obs1"])
            obs2.append(dict_line["obs2"])
            hyp1.append(dict_line["hyp1"])
            hyp2.append(dict_line["hyp2"])
    
    return story_id, obs1, obs2, hyp1, hyp2

def process_labels(file_path):
    labels = []
    with open(file_path) as file:
        for line in file:
            if line != " ":
                labels.append(line.replace("\n", ""))
    return labels

def create_csv(story_id, obs1, obs2, hyp1, hyp2, labels, file_path):
    rows = zip(story_id,obs1,obs2,hyp1,hyp2,labels)
    with open(file_path, "w") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)
