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
import ai.konduit.serving.util.ObjectMapperHolder;
import ai.konduit.serving.verticles.inference.InferenceVerticle;
import ai.konduit.serving.verticles.numpy.tensorflow.BaseMultiNumpyVerticalTest;
import com.jayway.restassured.specification.RequestSpecification;
import com.mashape.unirest.http.HttpResponse;
import com.mashape.unirest.http.JsonNode;
import com.mashape.unirest.http.Unirest;
import com.mashape.unirest.http.exceptions.UnirestException;
import io.vertx.core.AbstractVerticle;
import io.vertx.core.Handler;
import io.vertx.core.http.HttpClient;
import io.vertx.core.http.HttpServerRequest;
import io.vertx.core.json.Json;
import io.vertx.core.json.JsonArray;
import io.vertx.core.json.JsonObject;
import io.vertx.ext.unit.Async;
import io.vertx.ext.unit.TestContext;
import io.vertx.ext.unit.junit.VertxUnitRunner;
import jdk.nashorn.internal.parser.JSONParser;
import org.apache.commons.io.FileUtils;
import org.datavec.api.transform.schema.Schema;
import org.datavec.python.PythonVariables;
import org.junit.After;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;

import javax.annotation.concurrent.NotThreadSafe;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.charset.Charset;
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
public class KerasTestPythonJsonInputFormat extends BaseMultiNumpyVerticalTest {

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

        String pythonCodePath = new ClassPathResource("scripts/keras/KerasJsonTest.py").getFile().getAbsolutePath();

        PythonConfig pythonConfig = PythonConfig.builder()
                .pythonPath("C:\\Users\\Rabert-NIdrive\\AppData\\Local\\Programs\\Python\\Python37\\python37.zip;" +
                        "C:\\Users\\Rabert-NIdrive\\AppData\\Local\\Programs\\Python\\Python37\\DLLs" +
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
                        "C:\\Users\\Rabert-NIdrive\\AppData\\Local\\Programs\\Python\\Python37\\lib\\site-packages\\jnius-1.1.0-py3.7-win-amd64.egg;")

                .pythonCodePath(pythonCodePath)
                .pythonInput("JsonInput", PythonVariables.Type.STR.name())
                .pythonOutput("score", PythonVariables.Type.NDARRAY.name())
                .build();

        PythonStep pythonStepConfig = new PythonStep(pythonConfig);

        ServingConfig servingConfig = ServingConfig.builder()
                .httpPort(port)
                .inputDataFormat(Input.DataFormat.JSON)
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

        File json = new ClassPathResource("Json/IrisY.json").getFile();
        jsonObject.put("JsonInput", json.getAbsolutePath());
        requestSpecification.body(jsonObject.encode());

        requestSpecification.header("Content-Type", "application/json");
        String output = requestSpecification.when()
                .expect().statusCode(200)
                .body(not(isEmptyOrNullString()))
                .post("/raw/json").then()
                .extract()
                .body().asString();


        JsonObject jsonObject1 = new JsonObject(output);
        String ndarraySerde = jsonObject1.getJsonObject("default").toString();
        NDArrayOutput nd = ObjectMapperHolder.getJsonMapper().readValue(ndarraySerde, NDArrayOutput.class);
        INDArray outputArray = nd.getNdArray();
        INDArray expected = outputArray.get();
        assertEquals(expected, outputArray);

    }


}