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

package ai.konduit.serving.verticles.python.keras;

import ai.konduit.serving.InferenceConfiguration;
import ai.konduit.serving.config.Input;
import ai.konduit.serving.config.Output;
import ai.konduit.serving.config.ServingConfig;
import ai.konduit.serving.model.PythonConfig;
import ai.konduit.serving.output.types.NDArrayOutput;
import ai.konduit.serving.pipeline.step.PythonStep;
import ai.konduit.serving.verticles.inference.InferenceVerticle;
import ai.konduit.serving.verticles.numpy.tensorflow.BaseMultiNumpyVerticalTest;
import com.jayway.restassured.specification.RequestSpecification;
import com.mashape.unirest.http.Unirest;
import io.vertx.core.AbstractVerticle;
import io.vertx.core.Handler;
import io.vertx.core.http.HttpServerRequest;
import io.vertx.core.json.JsonObject;
import io.vertx.ext.unit.TestContext;
import io.vertx.ext.unit.junit.VertxUnitRunner;
import org.datavec.api.transform.schema.Schema;
import org.datavec.python.PythonVariables;
import org.junit.After;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.nd4j.jackson.objectmapper.holder.ObjectMapperHolder;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.serde.binary.BinarySerde;

import javax.annotation.concurrent.NotThreadSafe;
import java.io.File;
import java.util.Arrays;
import java.util.Objects;
import java.util.stream.Collectors;

import static com.jayway.restassured.RestAssured.given;
import static org.bytedeco.cpython.presets.python.cachePackages;

@RunWith(VertxUnitRunner.class)
@NotThreadSafe
public class KerasTestPythonNdArrayInputFormat extends BaseMultiNumpyVerticalTest {

    private Schema inputSchema;

    @Override
    public Class<? extends AbstractVerticle> getVerticalClazz() {
        return InferenceVerticle.class;
    }

    @After
    public void after(TestContext context) {
        vertx.close(context.asyncAssertSuccess());
    }

    @Override
    public Handler<HttpServerRequest> getRequest() {

        return req -> {
            //should be json body of classification
            req.bodyHandler(body -> {
                System.out.println(body.toJson());
                System.out.println("Finish body" + body);
            });

            req.exceptionHandler(exception -> context.fail(exception));
        };
    }

    @Override
    public JsonObject getConfigObject() throws Exception {
        String pythonPath = Arrays.stream(cachePackages())
                .filter(Objects::nonNull)
                .map(File::getAbsolutePath)
                .collect(Collectors.joining(File.pathSeparator));

/*

        String  pythonPath = "C:\\Users\\venkat-nidrive\\.javacpp\\cache\\cpython-3.7.3-1.5.1-windows-x86_64.jar\\org\\bytedeco\\cpython\\windows-x86_64\\\n" +
                "lib;C:\\Users\\venkat-nidrive\\.javacpp\\cache\\cpython-3.7.3-1.5.1-windows-x86_64.jar\\org\\bytedeco\\cpython\\windows-x86_64\\lib\\python3.7;\n" +
                "C:\\Users\\venkat-nidrive\\.javacpp\\cache\\cpython-3.7.3-1.5.1-windows-x86_64.jar\\org\\bytedeco\\cpython\\windows-x86_64\\lib\\python3.7\\lib-dynload;\n" +
                "C:\\Users\\venkat-nidrive\\.javacpp\\cache\\cpython-3.7.3-1.5.1-windows-x86_64.jar\\org\\bytedeco\\cpython\\windows-x86_64\\lib\\python3.7\\site-packages;\n" +
                "C:\\Users\\venkat-nidrive\\.javacpp\\cache\\numpy-1.16.4-1.5.1-windows-x86_64.jar\\org\\bytedeco\\numpy\\\n" +
                "windows-x86_64\\python";
*/

        String pythonCodePath = new ClassPathResource("scripts/keras/KerasNDArrayTest.py").getFile().getAbsolutePath();

        PythonConfig pythonConfig = PythonConfig.builder()
         //       .pythonPath(pythonPath)
                .pythonPath("C:\\Users\\venkat-nidrive\\AppData\\Local\\Programs\\Python\\Python37\\python37.zip;" +
                                "C:\\Users\\venkat-nidrive`AppData\\Local\\Programs\\Python\\Python37\\" +
                                "DLLs;C:\\Users\\venkat-nidrive\\AppData\\Local\\Programs\\Python\\Python37\\" +
                                "lib;C:\\Users\\venkat-nidrive\\AppData\\Local\\Programs\\Python\\Python37;" +
                                "C:\\Users\\venkat-nidrive\\AppData\\Local\\Programs\\Python\\Python37\\lib\\" +
                                "site-packages;C:\\Users\\venkat-nidrive\\AppData\\Local\\Programs\\Python\\Python37\\lib\\site-packages\\" +
                        "pyyaml-5.2-py3.7-win-amd64.egg;c:\\projects\\konduit-new\\konduit-serving\\python")

                .pythonCodePath(pythonCodePath)
                .pythonInput("default", PythonVariables.Type.NDARRAY.name())
                .pythonOutput("arr", PythonVariables.Type.NDARRAY.name())
                .build();

        PythonStep pythonStepConfig = new PythonStep(pythonConfig);

        ServingConfig servingConfig = ServingConfig.builder()
                .httpPort(port)
               // .inputDataFormat(Input.DataFormat.NUMPY)
                //.outputDataFormat(Output.DataFormat.NUMPY)
                .predictionType(Output.PredictionType.RAW)
                .build();

        InferenceConfiguration inferenceConfiguration = InferenceConfiguration.builder()
                .step(pythonStepConfig)
                .servingConfig(servingConfig)
                .build();

        return new JsonObject(inferenceConfiguration.toJson());
    }

    @Test(timeout = 60000)
    public void testInferenceResult(TestContext context) throws Exception {
        this.context = context;
        RequestSpecification requestSpecification = given();
        requestSpecification.port(port);
        JsonObject jsonObject = new JsonObject();

        //Preparing input NDArray
        INDArray arr = Nd4j.create(new float[]{1.0f, 2.0f,3.0f,4.0f }, 1, 4);

        String filePath = new ClassPathResource("data").getFile().getAbsolutePath();

        System.out.println("filePath-----------" + filePath);

        //Create new file to write binary input data.
        File file = new File(filePath + "/test-input.zip");
        System.out.println(file.getAbsolutePath());

        BinarySerde.writeArrayToDisk(arr, file);
        requestSpecification.body(jsonObject.encode().getBytes());

        requestSpecification.header("Content-Type", "multipart/form-data");
        String response = Unirest.post("http://localhost:" + port + "/raw/nd4j")
                .field("default", file)
                .asString().getBody();

        System.out.print(response);

        JsonObject jsonObject1 = new JsonObject(response);
        String ndarraySerde = jsonObject1.getJsonObject("default").toString();
        NDArrayOutput nd = ObjectMapperHolder.getJsonMapper().readValue(ndarraySerde, NDArrayOutput.class);

    }


}