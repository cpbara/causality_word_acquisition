from datetime import datetime

class Stopwatch:
    def __init__(self):
        self.base = datetime.now()
        self.mark = self.base
    
    def start(self):
        self.base = datetime.now()
        
    def lap(self):
        n = datetime.now()
        lap_diff = n - self.mark
        total_diff = n - self.base
        self.mark = n
        return lap_diff, total_diff
    
    def lap_str(self):
        ep, tot = self.lap()
        ep_text  = ':'.join([   f'{int(ep.total_seconds()//3600):1d}',
                                f'{int(ep.total_seconds()%3600//60):02d}',
                                f'{ep.total_seconds()%60:05.2f}'
                            ])
        tot_text = ':'.join([   f'{int(tot.total_seconds()//3600):3d}',
                                f'{int(tot.total_seconds()%3600//60):02d}',
                                f'{tot.total_seconds()%60:05.2f}'
                            ])
        return ' '.join([ep_text, tot_text])