class Sample:
    def __init__(self, text, audio_path, label, dialogue_id, speaker_id, start, end, prev_end, embeddings, mfcc):
        self.text = text
        self.audio_path = audio_path
        self.label = label
        self.dialogue_id = dialogue_id
        self.speaker_id = speaker_id
        self.start = start
        self.end = end
        self.pause = start - prev_end if prev_end else 0
        self.embeddings = embeddings
        self.mfcc = mfcc