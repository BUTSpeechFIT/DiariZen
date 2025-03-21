from diarizen.pipelines.inference import DiariZenPipeline

# load pre-trained model
diar_pipeline = DiariZenPipeline.from_pretrained("BUT-FIT/diarizen-meeting-base")

# apply diarization pipeline
diar_results = diar_pipeline('EN2002a_30s.wav')

# print results
for turn, _, speaker in diar_results.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
# start=0.0s stop=1.7s speaker_0
# start=0.0s stop=0.7s speaker_1
# start=0.8s stop=8.0s speaker_4
# ...

# load pre-trained model and save RTTM result
diar_pipeline = DiariZenPipeline.from_pretrained(
        "BUT-FIT/diarizen-meeting-base",
        rttm_out_dir='.'
)
# apply diarization pipeline
diar_results = diar_pipeline('EN2002a_30s.wav', sess_name='EN2002a')