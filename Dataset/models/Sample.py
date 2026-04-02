class Sample:
    def __init__(
        self,
        text,
        audio_path,
        label,
        dialogue_id,
        start,
        end,
        prev_end,
        embeddings,
        audio_features,
        speaker_name=None,
        speaker_id=None,
        pause_norm_mu=None,
        pause_norm_std=None,
    ):
        self.text = text
        self.audio_path = audio_path
        self.label = label
        self.dialogue_id = dialogue_id
        self.speaker_name = speaker_name
        self.speaker_id = speaker_id
        self.start = start
        self.end = end
        self.pause = start - prev_end if prev_end else 0
        self.embeddings = embeddings
        self.audio_features = audio_features
        # z-score(log1p pause)) по train; выставляется в preprocess для всех сплитов
        self.pause_norm_mu = pause_norm_mu
        self.pause_norm_std = pause_norm_std
