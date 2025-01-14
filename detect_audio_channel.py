import pyaudio

p = pyaudio.PyAudio()

print("Dispositivos disponibles:")
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    print(f"Índice {i}: {info['name']}")
p.terminate()
