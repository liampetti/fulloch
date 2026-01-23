import wave, struct, math

sample_rate = 44100
beep_freq = 880.0
beep_ms = 120
gap_ms = 80
repeats = 5
amplitude = 0.45

def ramp_env(i, total):
    ramp = int(0.015 * sample_rate)  # 15 ms ramp
    a = min(1.0, i / max(1, ramp), (total - i) / max(1, ramp))
    return a

def tone_samples(freq, ms):
    n = int(sample_rate * (ms/1000.0))
    for i in range(n):
        t = i / sample_rate
        yield amplitude * ramp_env(i, n) * math.sin(2*math.pi*freq*t)

def silence_samples(ms):
    n = int(sample_rate * (ms/1000.0))
    for _ in range(n):
        yield 0.0

with wave.open("alarm.wav", "w") as wf:
    wf.setnchannels(1)       # mono
    wf.setsampwidth(2)       # 16-bit
    wf.setframerate(sample_rate)

    frames = []
    for r in range(repeats):
        frames.extend(tone_samples(beep_freq, beep_ms))
        if r != repeats - 1:
            frames.extend(silence_samples(gap_ms))

    # write frames
    wf.writeframes(b"".join(struct.pack("<h", int(max(-1.0, min(1.0, x)) * 32767)) for x in frames))

print("Wrote alarm.wav")
