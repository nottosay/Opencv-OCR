package com.opencv.demo;

import java.util.List;

/**
 * Created by wally.yan on 2016/10/25.
 */

public class OpenCVHelper {
    static{
        System.loadLibrary("OpenCV");
    }

    public static native List<String> ocr(String imgPach,String intermediatePach);
}
