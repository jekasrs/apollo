class Sample:
    def __init__(self, utterance_id, text,
        audio_path, label, dialogue_id, start, end, embeddings,
        audio_features, speaker_name, pause=0.0,
        speaker_id=None, pause_norm_mu=None, pause_norm_std=None,
    ):
        self.utterance_id = utterance_id
        self.text = text
        self.audio_path = audio_path
        self.label = label
        self.dialogue_id = dialogue_id
        self.speaker_name = speaker_name
        self.speaker_id = speaker_id
        self.start = start
        self.end = end
        self.pause = float(pause) if pause is not None else 0.0
        self.embeddings = embeddings
        self.audio_features = audio_features
        self.pause_norm_mu = pause_norm_mu
        self.pause_norm_std = pause_norm_std
