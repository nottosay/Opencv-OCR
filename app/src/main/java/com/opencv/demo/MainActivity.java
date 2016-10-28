package com.opencv.demo;

import android.app.Activity;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Message;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

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
                new Mythread().start();
            }
        });
    }

    Handler handler = new Handler() {
        @Override
        public void handleMessage(Message msg) {
            super.handleMessage(msg);
            if (1 == msg.what) {
                String s = msg.getData().getString("s");
                textView.setText(s);
            }
        }
    };


    class Mythread extends Thread {
        @Override
        public void run() {
            String imgUrl = Environment.getExternalStorageDirectory() + "/id_card.jpg";
            Log.i("wally", imgUrl);
            String s = OpenCVHelper.ocr(imgUrl);
            Message message = handler.obtainMessage();
            message.what = 1;
            Bundle b = new Bundle();
            b.putString("s", s);
            message.setData(b);
            handler.sendMessage(message);
        }
    }
}
