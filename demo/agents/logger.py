import os.path
from dataclasses import dataclass


@dataclass
class Log:
    entity: str
    content: str = ""
    df_sample: str = None

    def append_content(self, title, content):
        self.content += f"\n\n{title}\n\n"
        self.content += "="*50
        self.content += "\n\n"
        self.content += content

    def set_df_sample(self, df):
        self.df_sample = df.head(10).to_string()


class Logger:
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    cyan = "\x1b[36;20m"
    magenta = "\x1b[35;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    FORMATS = {
        "Default": f"{grey}msg_content{reset}",
        "Actor": f"{cyan}msg_content{reset}",
        "Critic": f"{magenta}msg_content{reset}",
    }

    def __init__(self, task_id, original_operation, output_path_prefix="logs"):
        self.logs = []
        self.task_id = task_id
        self.original_operation = original_operation
        self.output_folder_path = os.path.join(output_path_prefix, task_id)

    def log(self, log: Log):
        self.logs.append(log)

    def get_logs(self):
        return self.logs

    def save_logs(self):
        os.makedirs(self.output_folder_path, exist_ok=True)
        with open(os.path.join(self.output_folder_path, f"{self.task_id}.log"), "w") as f:
            for log in self.logs:
                frmt = self.FORMATS.get(log.entity, self.FORMATS["Default"])
                f.write(frmt.replace("msg_content", "="*50))
                f.write(frmt.replace("msg_content", f"{log.entity}:"))
                f.write(frmt.replace("msg_content", "="*50))
                f.write("\n\n")
                f.write(frmt.replace("msg_content", f"{log.content}"))
                f.write("\n\n")
        saved_to = os.path.join(self.output_folder_path, f"{self.task_id}.log")
        print(f"Logs saved at {saved_to}")