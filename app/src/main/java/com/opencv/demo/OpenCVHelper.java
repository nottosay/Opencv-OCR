package com.opencv.demo;

/**
 * Created by wally.yan on 2016/10/25.
 */

public class OpenCVHelper {
    static{
        System.loadLibrary("OpenCV");
    }

    public static native String ocr(String str);
}
