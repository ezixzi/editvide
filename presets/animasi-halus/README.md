# Animasi Halus

Preset produksi Reels 3D untuk gaya Animasi Halus. Pakai untuk permintaan Animasi Halus dan episode Lah Kok Bisa yang memakai gaya ini. Susun lokasi, aksi, dan blocking kamera dari naskah episode.

## Gaya

- Karakter 3D stylized dengan wajah dan profil kostum yang konsisten. Untuk Zii, gunakan rambut hitam pendek dan kaus navy. Tangan close-up memakai lima jari dan permukaan skinned, lengan membengkok sebagai satu bentuk. Mulai aksi dari tatapan atau perubahan tumpuan, beri waktu untuk mengendap sesudahnya.
- Bentuk ruang dengan furnitur dan properti yang berhubungan dengan cerita. Pilih sudut lebar untuk lokasi, medium untuk interaksi, dan dekat untuk reaksi atau detail. Gunakan perpindahan kamera yang termotivasi, cahaya hangat dan sejuk sesuai waktu, serta fokus yang mengikuti kedalaman.
- Caption masuk per kata dengan fade dan perubahan ukuran kecil. Sisakan area wajah dan tangan. Gunakan outline tipis dan aksen lime secukupnya. Koordinat pada `profile.json` memakai kanvas 1080 × 1920.
- Pakai VO asli sebagai durasi dan acuan setiap cut. Musik instrumental dan Foley mengikuti adegan dengan ducking saat narasi terdengar. SFX ditempatkan pada kejadian yang bisa dilihat atau didengar dalam cerita.

## Masukan per episode

Simpan `scene.py`, `timeline.json`, `captions.json`, dan VO di `projects/<episode>`. Pisahkan aset dan output episode dari preset.

| Masukan | Isi |
| --- | --- |
| `timeline.json` | `title`, jumlah `frames`, durasi VO asli `source_duration`, dan `shots` dengan `id`, `start`, `end` dalam frame. Mulai dari frame 0. `end` eksklusif. Gunakan `ceil(source_duration * 30)` untuk jumlah frame. |
| `captions.json` | Daftar cue dengan `start`, `end` dalam detik dan satu atau dua `lines`. Setiap kata pada baris punya `text` dan `onset` dalam detik. Cocokkan teks dan timing dengan VO. |
| `scene.py` | `atlas()` mengembalikan gambar RGBA dan rect UV. `install(stage)` mendaftarkan mesh khusus. `draw(film, shot, g, shadow, air, t, p)` mengembalikan `(world, (eye, aim, fov), light, focus)`. `labels(shot, t, p)` memberi baris `(text, x, y, size, opacity, accent)`. |
| Aksen dan Foley | `keywords` opsional mewarnai kata tertentu. `caption_y` opsional memilih posisi vertikal tiap shot. `effects` opsional berisi `[detik, jenis, durasi, gain, pan]`. Pan −1 hingga 1. |

Gunakan fungsi register karakter di `cast.py` untuk warna kulit, rambut, pakaian, dan ukuran. Buat set dengan `studio.py` dan cache set tetap melalui `film.world()`. Gunakan `focus` sebagai titik dunia yang dipertahankan tajam. Simpan lintasan dan keputusan episode di modul adegannya.

## Jalankan dari akar repo

Butuh Python 3.11+, NumPy, SciPy, Pillow, FFmpeg dengan libx264, serta OpenGL 3.3 dan EGL surfaceless. Renderer dapat berjalan lewat Mesa llvmpipe di CPU. Blender diperlukan saat membuat ulang mesh dari `model_hands.py`. Untuk render rutin, `hands.npz` sudah tersedia.

```bash
preset=presets/animasi-halus
episode=projects/nama-episode
work="$episode/artifacts"
export LP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1

python3 "$preset/render.py" --episode "$episode" --out "$work/proof" --preview --stills 12 38
python3 "$preset/render.py" --episode "$episode" --out "$work/proof" --preview --audit
python3 "$preset/render.py" --episode "$episode" --out "$work/plates"
python3 "$preset/audio.py" --voice "$episode/assets/audio/vo.wav" --timeline "$episode/timeline.json" --out "$work/audio"
python3 "$preset/export.py" --episode "$episode" --plates "$work/plates" --audio "$work/audio/mix-master.wav" --work "$work/export" --out "output/nama-episode/FINAL.mp4"
```

## Pemeriksaan per episode

1. Pasang `film.track()` pada kepala atau objek utama yang harus utuh. Pasang `film.clearance()` pada pasangan yang berpotensi bertabrakan dan `film.smooth()` pada gerak yang rawan meloncat. `contacts.py` memberi posisi tangan terhadap bidang kerja dan mengecek mesh tangan yang dipakai saat render.
2. Render uji sebelum, saat, dan sesudah kontak. Periksa jari, tumpuan kaki, lipatan kain, objek di tangan, posisi kepala, kamera, dan caption. Uji terutama perubahan posisi tangan terhadap meja, kaca, kain atau panel yang bergerak. Pemeriksaan numerik mencakup lintasan yang diberi batas di modul adegan.
3. Perbaiki pose dan render ulang shot bermasalah dengan `--shots` menggunakan indeks mulai dari nol. Ekspor memeriksa semua frame, dimensi, audio, posisi frame, black frame dan freeze. Buka MP4 gabungan dan periksa transisi serta kontak yang paling berisiko.
4. Simpan video final ke Dropbox `/Video Output` dan verifikasi ukuran serta tautannya. Simpan kode preset di repo.
