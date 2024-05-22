from phonemizer.separator import Separator
from phonemizer.backend import EspeakBackend

backend = EspeakBackend(
    language='es-419',
    with_stress=True,
    separator=Separator(phone='-', word=' ')
)

def rate_apply(batch, rank=None, audio_column_name="audio", text_column_name="text"):
    """
    Calculate speaking rates and phonemes for audio-text pairs in a batch.

    Parameters:
    - batch (dict): The batch of data containing audio and text fields.
    - rank (optional): The rank of processing unit in distributed computing.
    - audio_column_name (str): The key in the batch dict that refers to the audio data.
    - text_column_name (str): The key in the batch dict that refers to the text data.

    Returns:
    - dict: The updated batch including speaking rates and phonemized text.
    """
    # Instantiate the backend once
    def phonemize_texts(texts):
        return [backend.phonemize(text) for text in texts]

    if isinstance(batch[audio_column_name], list):
        phonemes_list = phonemize_texts(batch[text_column_name])
        
        speaking_rates = []
        for phonemes, audio in zip(phonemes_list, batch[audio_column_name]):
            sample_rate = audio.get("sampling_rate")
            audio_array = audio.get("array")
            if sample_rate is None or audio_array is None:
                continue

            audio_length = len(audio_array.squeeze()) / sample_rate
            speaking_rate = len(phonemes) / audio_length

            speaking_rates.append(speaking_rate)
        
        batch["speaking_rate"] = speaking_rates
        batch["phonemes"] = phonemes_list
    else:
        phonemes = backend.phonemize(batch[text_column_name])
        
        sample_rate = batch[audio_column_name].get("sampling_rate")
        audio_array = batch[audio_column_name].get("array")
        if sample_rate is None or audio_array is None:
            return batch

        audio_length = len(audio_array.squeeze()) / sample_rate
        speaking_rate = len(phonemes) / audio_length
        
        batch["speaking_rate"] = speaking_rate
        batch["phonemes"] = phonemes

    return batch
