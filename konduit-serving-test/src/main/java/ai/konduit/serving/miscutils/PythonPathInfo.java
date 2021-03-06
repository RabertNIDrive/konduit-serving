/*
 *
 *  * ******************************************************************************
 *  *  * Copyright (c) 2015-2019 Skymind Inc.
 *  *  * Copyright (c) 2019 Konduit AI.
 *  *  *
 *  *  * This program and the accompanying materials are made available under the
 *  *  * terms of the Apache License, Version 2.0 which is available at
 *  *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *  *
 *  *  * Unless required by applicable law or agreed to in writing, software
 *  *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  *  * License for the specific language governing permissions and limitations
 *  *  * under the License.
 *  *  *
 *  *  * SPDX-License-Identifier: Apache-2.0
 *  *  *****************************************************************************
 *
 *
 */
package ai.konduit.serving.miscutils;

import java.io.BufferedReader;
import java.io.InputStreamReader;

public class PythonPathInfo {

    public PythonPathInfo() {
    }

    public static String getPythonPath()
            throws Exception {

        //To detect host OS.
        String operSys = System.getProperty("os.name").toLowerCase();

        //To get local Python path run the below command and get the path for python libraries.
        String[] args;
        if (operSys.indexOf("win") >= 0) {
            args = new String[] {"cmd.exe", "/c", "python -c \"import sys, os; print(os.pathsep.join([path for path in sys.path if path]))\"" , "with", "args"};
        } else if (operSys.indexOf("nix") >= 0 || operSys.indexOf("nux") >= 0 || operSys.indexOf("aix") >= 0) {
            args = new String[] {"/bin/bash", "-c", " python3.7 -c \"import os, sys; print(os.path.pathsep.join([path.strip() for path in sys.path if path.strip()]))\"" , "with", "args"};
        } else{
            args = new String[] {"cmd.exe", "/c", "python -c \"import sys, os; print(os.pathsep.join([path for path in sys.path if path]))\"" , "with", "args"};
        }

        ProcessBuilder builder = new ProcessBuilder(args);
        builder.redirectErrorStream(true);
        Process p = builder.start();
        BufferedReader r = new BufferedReader(new InputStreamReader(p.getInputStream()));
        String line;
        String pythonPath = "";
        while (true) {
            line = r.readLine();
            if (line == null) {
                break;
            }

            //Confirm the path
            System.out.println(line);
            pythonPath = line;

        }

        return pythonPath;

        //
        //Python path to get from the JavaCPP
        //        String pythonPath = Arrays.stream(cachePackages())
        //                .filter(Objects::nonNull)
        //                .map(File::getAbsolutePath)
        //                .collect(Collectors.joining(File.pathSeparator));
        //


    }


}
