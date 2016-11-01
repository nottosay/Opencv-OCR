package com.opencv.demo;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import com.googlecode.tesseract.android.TessBaseAPI;

import java.util.List;

import rx.Observable;
import rx.Subscriber;
import rx.android.schedulers.AndroidSchedulers;
import rx.functions.Func0;
import rx.schedulers.Schedulers;

public class MainActivity extends Activity {

    private TextView textView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button button = (Button) findViewById(R.id.btn);
        textView = (TextView) findViewById(R.id.tv);
        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                toOcr();
            }
        });
    }


    private void toOcr() {
        Observable.defer(new Func0<Observable<List<String>>>() {
            @Override
            public Observable<List<String>> call() {
                String imgUrl = Environment.getExternalStorageDirectory() + "/id_card.jpg";
                Log.i("wally", imgUrl);
                List<String> s = OpenCVHelper.ocr(imgUrl, getFilesDir() + "/opecv_img/");
                return Observable.just(s);
            }
        }).subscribeOn(Schedulers.io()) // 指定 subscribe() 发生在 IO 线程
                .observeOn(AndroidSchedulers.mainThread()) // 指定 Subscriber 的回调发生在主线程
                .subscribe(new Subscriber<List<String>>() {
                    @Override
                    public void onCompleted() {

                    }

                    @Override
                    public void onError(Throwable e) {

                    }

                    @Override
                    public void onNext(List<String> strings) {
                        for (String s : strings) {
                            Log.i("wally", s);
                        }
                    }
                });
    }


    /**
     * 进行图片识别
     *
     * @param fileName 待识别图片
     * @return 识别结果字符串
     */
    public String doOcr(String fileName) {

        Bitmap bitmap = BitmapFactory.decodeFile(Environment.getExternalStorageDirectory() + "/opencv_img/" + fileName);

        TessBaseAPI baseApi = new TessBaseAPI();

        baseApi.init(Environment.getExternalStorageDirectory().getPath(), "/chi_sim");

        // 必须加此行，tess-two要求BMP必须为此配置
        bitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);

        baseApi.setImage(bitmap);

        String text = baseApi.getUTF8Text();

        baseApi.clear();
        baseApi.end();

        return text;
    }
}
