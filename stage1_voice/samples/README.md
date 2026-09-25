# stage1_voice/samples

Record the official challenge command as `official_command.wav` in this folder:

> "Open the top drawer, pick up the plate with arm A, place it on the table,
> pick up the mug with arm B, pour water into the mug with arm A."

It is used as the test fixture for `parse_command` and later in the demo:

```bash
export SPEECHMATICS_API_KEY=...   # and ANTHROPIC_API_KEY
python -m stage1_voice --audio stage1_voice/samples/official_command.wav
```

Any common format works (wav/mp3/m4a); mono 16 kHz wav is a safe choice.
