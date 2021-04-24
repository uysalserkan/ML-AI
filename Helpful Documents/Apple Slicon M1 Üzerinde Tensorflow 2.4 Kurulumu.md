# Apple Silicon M1 üzerinde Conda yardımıyla Tensorflow 2.4 kurulumu

![apple_silicon](/images/apple-silicon.jpg)

Bu makalede Tensorflow 2.4 kullanımı için M1 işlemciler için aktif olan **ATF 2.4** kullanılacaktır. Şu an için [github adresinde](https://github.com/apple/tensorflow_macos) 0.1 sürümü aktif halde indirilebilir durumdadır.

*Paket hala geliştirilme aşamasında olduğu için içerisinde bir çok hata bulundurabilir. Her kurulum farklı hatalar ile karşılaşabilirsiniz. Profesyonel kullanım yerine sadece M1 işlemciler üzerinde Tensorflow 2.4 deneyimlemek için kurulumu gerçekleştirirseniz daha güvenli bir yol tercih etmiş olursunuz.*

Conda üzerinden kurulacak paketler:

* ATF 2.4 (Apple Silicon için Tensorflow 2.4)
* Numpy
* Scikit-learn
* Pandas
* Matplotlib
* Jupyter Lab

## Adım 1: Xcode Komut Satırı Araçları

[Xcode Command Line Tools](https://developer.apple.com/download/more/?=command%20line%20tools)'u direkt olarak [Apple Developer](https://developer.apple.com/) sitesinden indirebileceğiniz gibi, aşağıdaki komut satırını çalıştırarak da kurabilirsiniz.

```shell
xcode-select --install
```

## Adım 2: miniforge

[Miniforge github](https://github.com/conda-forge/miniforge) repositorisinden  Apple Silicon için direkt olarak indirip kurulumunu gerçekleştirebilirsiniz.
Miniforge python paketlerini Apple Silicon üzerinden *native* olarak çalışıtırlmasını sağlayan bir arçatır.

## Adım 3: ATF 2.4 indirilmesi

[Apple'ın resmi github repositorisinden](https://github.com/apple/tensorflow_macos) direkt olarak indirip *untar* işlemini gerçekleştiriniz, **bu aşamada standart yükleyici ile yüklemeyin**.

Çıkarttığınız dosyanın içerisinde bulunan `arm64` klasörüne gidiniz;

```shell
cd tensorflow_macos/arm64
```

## Adım 4: Conda üzerinde ortam oluşturma

Miniforge kurduktan sonra yeni bir *sezon* başlatmayı unutmayın. (Bence siz en iyisi bilgisayarı yeniden başlatın. :) )

Sonraki aşama olarak Conda üzerinde **yeni bir ortam** oluşturunuz, bu ortamın Python sürümü **3.8** olduğuna emin olunuz. (*Tensorflow 2.4 Python 3.8 üzerinde çalışıyor*)

Aşağıdaki kod parçası ile yukarıda belirtilen işlemleri temrinal üzerinden gerçekleştirebilirsiniz;

```shell
conda create --name tf24
conda activate tf24
conda install -y python==3.8.6
conda install -y pandas matplotlib scikit-learn jupyterlab
```

## Adım 5: ATF 2.4 Kurulumu

Conda ortamınız üzerinde `install_venv.sh` dosyasını çalıştırarak **ATF 2.4** kurulumunu gerçekleştirebilirsiniz.

install_venv.sh içeriğinin aşağıdaki kod parçası olduğunun doğruluğunu kontrol edin;

```shell
# Install specific pip version and some other base packages
pip install --force pip==20.2.4 wheel setuptools cached-property six
# Install all the packages provided by Apple but TensorFlow
pip install --upgrade --no-dependencies --force numpy-1.18.5-cp38-cp38-macosx_11_0_arm64.whl grpcio-1.33.2-cp38-cp38-macosx_11_0_arm64.whl h5py-2.10.0-cp38-cp38-macosx_11_0_arm64.whl tensorflow_addons-0.11.2+mlcompute-cp38-cp38-macosx_11_0_arm64.whl
# Install additional packages
pip install absl-py astunparse flatbuffers gast google_pasta keras_preprocessing opt_einsum protobuf tensorflow_estimator termcolor typing_extensions wrapt wheel tensorboard typeguard
# Install TensorFlow
pip install --upgrade --force --no-dependencies tensorflow_macos-0.1a1-cp38-cp38-macosx_11_0_arm64.whl
```

### Her hangi bir sorunla karşılaşmanız durumunda repository üzerinden konu açmayı ihmal etmeyin

###### [Kaynak](https://towardsdatascience.com/tensorflow-2-4-on-apple-silicon-m1-installation-under-conda-environment-ba6de962b3b8)
