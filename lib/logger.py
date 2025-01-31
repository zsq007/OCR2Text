import os
import csv

class CSVLogger:

    def __init__(self, name, path, fieldnames):
        self.log_file = open(os.path.join(path, name), 'w', newline='')
        self.writer = csv.DictWriter(self.log_file, fieldnames)
        self.writer.writeheader()

    def update_with_dict(self, dict_entries):
        self.writer.writerow(dict_entries)
        self.log_file.flush()

    def update_with_dicts(self, list_dicts):
        self.writer.writerows(list_dicts)
        self.log_file.flush()

    def close(self):
        del self.writer
        self.log_file.close()