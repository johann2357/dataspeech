from phonemizer import phonemize
from phonemizer.separator import Separator
from phonemizer.backend import EspeakBackend

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
    def phonemize_text(text):
        return phonemize(
            text,
            language='es-419',
            backend='espeak',
            with_stress=True,
            separator=Separator(phone='-', word=' '),
            njobs=1
        )

    if isinstance(batch[audio_column_name], list):
        speaking_rates = []
        phonemes_list = []
        for text, audio in zip(batch[text_column_name], batch[audio_column_name]):
            phonemes = phonemize_text(text)
            
            sample_rate = audio.get("sampling_rate")
            audio_array = audio.get("array")
            if sample_rate is None or audio_array is None:
                continue

            audio_length = len(audio_array.squeeze()) / sample_rate
            speaking_rate = len(phonemes) / audio_length

            speaking_rates.append(speaking_rate)
            phonemes_list.append(phonemes)
        
        batch["speaking_rate"] = speaking_rates
        batch["phonemes"] = phonemes_list
    else:
        phonemes = phonemize_text(batch[text_column_name])
        
        sample_rate = batch[audio_column_name].get("sampling_rate")
        audio_array = batch[audio_column_name].get("array")
        if sample_rate is None or audio_array is None:
            return batch

        audio_length = len(audio_array.squeeze()) / sample_rate
        speaking_rate = len(phonemes) / audio_length
        
        batch["speaking_rate"] = speaking_rate
        batch["phonemes"] = phonemes

    return batch
