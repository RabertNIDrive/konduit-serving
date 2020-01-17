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
import ai.konduit.serving.config.Output;
import ai.konduit.serving.config.ServingConfig;
import ai.konduit.serving.model.PythonConfig;
import ai.konduit.serving.output.types.NDArrayOutput;
import ai.konduit.serving.pipeline.step.PythonStep;
import ai.konduit.serving.util.ObjectMapperHolder;
import ai.konduit.serving.verticles.inference.InferenceVerticle;
import ai.konduit.serving.verticles.numpy.tensorflow.BaseMultiNumpyVerticalTest;
import com.jayway.restassured.specification.RequestSpecification;
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
import static org.hamcrest.Matchers.isEmptyOrNullString;
import static org.hamcrest.Matchers.not;
import static org.junit.Assert.assertEquals;

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
        String pythonCodePath = new ClassPathResource("scripts/keras/KerasNDArrayTest.py").getFile().getAbsolutePath();

        PythonConfig pythonConfig = PythonConfig.builder()
                .pythonPath("C:\\Users\\Rabert-NIdrive\\AppData\\Local\\Programs\\Python\\Python37\\python37.zip;" +
                        "C:\\Users\\Rabert-NIdrive\\AppData\\Local\\Programs\\Python\\Python37\\DLLs;" +
                        "C:\\Users\\Rabert-NIdrive\\AppData\\Local\\Programs\\Python\\Python37\\lib;" +
                        "C:\\Users\\Rabert-NIdrive\\AppData\\Local\\Programs\\Python\\Python37;" +
                        "C:\\Users\\Rabert-NIdrive\\AppData\\Roaming\\Python\\Python37\\site-packages;" +
                        "C:\\Users\\Rabert-NIdrive\\AppData\\Roaming\\Python\\Python37\\site-packages\\win32;" +
                        "C:\\Users\\Rabert-NIdrive\\AppData\\Roaming\\Python\\Python37\\site-packages\\win32\\lib;" +
                        "C:\\Users\\Rabert-NIdrive\\AppData\\Roaming\\Python\\Python37\\site-packages\\Pythonwin;" +
                        "C:\\Users\\Rabert-NIdrive\\AppData\\Local\\Programs\\Python\\Python37\\lib\\site-packages;" +
                        "C:\\Users\\Rabert-NIdrive\\AppData\\Local\\Programs\\Python\\Python37\\lib\\site-packages\\pyyaml-5.2-py3.7-win-amd64.egg;" +
                        "C:\\Users\\Rabert-NIdrive\\AppData\\Local\\Programs\\Python\\Python37\\lib\\site-packages\\click-7.0-py3.7.egg;" +
                        "C:\\Users\\Rabert-NIdrive\\AppData\\Local\\Programs\\Python\\Python37\\lib\\site-packages\\pydatavec-0.1.2-py3.7.egg;" +
                        "C:\\Users\\Rabert-NIdrive\\AppData\\Local\\Programs\\Python\\Python37\\lib\\site-packages\\pydl4j-0.1.4-py3.7.egg;" +
                        "C:\\Users\\Rabert-NIdrive\\AppData\\Local\\Programs\\Python\\Python37\\lib\\site-packages\\cython-0.29.14-py3.7-win-amd64.egg;" +
                        "C:\\Users\\Rabert-NIdrive\\AppData\\Local\\Programs\\Python\\Python37\\lib\\site-packages\\pandas-0.24.2-py3.7-win-amd64.egg;" +
                        "C:\\Users\\Rabert-NIdrive\\AppData\\Local\\Programs\\Python\\Python37\\lib\\site-packages\\requests_toolbelt-0.9.1-py3.7.egg;" +
                        "C:\\Users\\Rabert-NIdrive\\AppData\\Local\\Programs\\Python\\Python37\\lib\\site-packages\\pyarrow-0.13.0-py3.7-win-amd64.egg;" +
                        "C:\\Users\\Rabert-NIdrive\\AppData\\Local\\Programs\\Python\\Python37\\lib\\site-packages\\numpy-1.16.4-py3.7-win-amd64.egg;" +
                        "C:\\Users\\Rabert-NIdrive\\AppData\\Local\\Programs\\Python\\Python37\\lib\\site-packages\\requests-2.22.0-py3.7.egg;" +
                        "C:\\Users\\Rabert-NIdrive\\AppData\\Local\\Programs\\Python\\Python37\\lib\\site-packages\\python_dateutil-2.8.1-py3.7.egg;" +
                        "C:\\Users\\Rabert-NIdrive\\AppData\\Local\\Programs\\Python\\Python37\\lib\\site-packages\\jnius-1.1.0-py3.7-win-amd64.egg;" +
                        "D:\\\\KonduitFork\\\\konduit-serving\\\\python")
                .pythonCodePath(pythonCodePath)
                .pythonInput("default", PythonVariables.Type.NDARRAY.name())
                .pythonOutput("arr", PythonVariables.Type.NDARRAY.name())
                .build();

        PythonStep pythonStepConfig = new PythonStep(pythonConfig);

        ServingConfig servingConfig = ServingConfig.builder()
                .httpPort(port)
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
        INDArray arr = Nd4j.create(new float[][]{{1, 0, 5, 10}, {100, 55, 555, 1000}});

        String filePath = new ClassPathResource("data").getFile().getAbsolutePath();

        //Create new file to write binary input data.
        File file = new File(filePath + "/test-input.zip");
        System.out.println(file.getAbsolutePath());

        BinarySerde.writeArrayToDisk(arr, file);
        requestSpecification.body(jsonObject.encode().getBytes());

        requestSpecification.header("Content-Type", "multipart/form-data");
        String response = requestSpecification.when()
                .multiPart("default", file)
                .expect().statusCode(200)
                .body(not(isEmptyOrNullString()))
                .post("/raw/nd4j").then()
                .extract()
                .body().asString();

        JsonObject jsonObject1 = new JsonObject(response);
        String ndarraySerde = jsonObject1.getJsonObject("default").toString();
        NDArrayOutput nd = ObjectMapperHolder.getJsonMapper().readValue(ndarraySerde, NDArrayOutput.class);
        INDArray outputArray = nd.getNdArray();
        INDArray expected = Nd4j.create(new float[][]{{0.1628401f, 0.7828045f, 0.05435541f}, {0.0f, 1.0f, 0.0f}});
        assertEquals(expected, outputArray);

    }


}
