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

package ai.konduit.serving.verticles.python.scikitlearn;

import ai.konduit.serving.InferenceConfiguration;
import ai.konduit.serving.config.Output;
import ai.konduit.serving.config.ServingConfig;
import ai.konduit.serving.miscutils.ExpectedAssertUtil;
import ai.konduit.serving.miscutils.PythonPathUtils;
import ai.konduit.serving.model.PythonConfig;
import ai.konduit.serving.pipeline.step.PythonStep;
import ai.konduit.serving.verticles.inference.InferenceVerticle;
import ai.konduit.serving.verticles.numpy.tensorflow.BaseMultiNumpyVerticalTest;
import com.jayway.restassured.specification.RequestSpecification;
import io.vertx.core.AbstractVerticle;
import io.vertx.core.Handler;
import io.vertx.core.http.HttpServerRequest;
import io.vertx.core.json.JsonObject;
import io.vertx.ext.unit.TestContext;
import io.vertx.ext.unit.junit.VertxUnitRunner;
import org.apache.commons.io.FileUtils;
import org.datavec.python.PythonType;
import org.junit.Assert;
import org.junit.Ignore;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.serde.binary.BinarySerde;

import javax.annotation.concurrent.NotThreadSafe;
import java.io.File;
import java.nio.charset.Charset;

import static com.jayway.restassured.RestAssured.given;
import static org.hamcrest.Matchers.isEmptyOrNullString;
import static org.hamcrest.Matchers.not;

@RunWith(VertxUnitRunner.class)
@NotThreadSafe
@Ignore
public class ScikitLearnPythonNd4jNd4jRegressionTest extends BaseMultiNumpyVerticalTest {

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    @Override
    public Class<? extends AbstractVerticle> getVerticalClazz() {
        return InferenceVerticle.class;
    }

    @Override
    public Handler<HttpServerRequest> getRequest() {

        return req -> {
            //should be json body of classification
            req.bodyHandler(body -> {
            });

            req.exceptionHandler(exception -> context.fail(exception));
        };
    }

    @Override
    public JsonObject getConfigObject() throws Exception {

        String pythonCodePath = new ClassPathResource("scripts/scikitlearn/ScikitLearnRegression.py").getFile().getAbsolutePath();

        PythonConfig pythonConfig = PythonConfig.builder()
                .pythonPath(PythonPathUtils.getPythonPath())
                .pythonCodePath(pythonCodePath)
                .pythonInput("inputData", PythonType.TypeName.NDARRAY.name())
                .pythonOutput("pred", PythonType.TypeName.NDARRAY.name())
                .build();

        PythonStep pythonStepConfig = new PythonStep(pythonConfig);

        ServingConfig servingConfig = ServingConfig.builder()
                .outputDataFormat(Output.DataFormat.ND4J)
                .httpPort(port)
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
        INDArray arr = Nd4j.create(new double[]{1462.00000,20.0000000,81.0000000,14267.0000,6.00000000,6.00000000,1958.00000,1958.00000,108.000000,923.000000,0.00000000,406.000000,1329.00000,1329.00000,0.00000000,0.00000000 ,1329.00000,0.00000000,0.00000000,1.00000000, 1.00000000,3.00000000,1.00000000,6.00000000, 0.00000000,1958.00000,1.00000000,312.000000, 393.000000,36.0000000,0.00000000,0.00000000, 0.00000000,0.00000000,12500.0000,6.00000000, 2010.00000,0.00000000,0.00000000,0.00000000, 1.00000000,0.00000000,0.00000000,1.00000000, 0.00000000,0.00000000,1.00000000,0.00000000, 0.00000000,0.00000000,0.00000000,0.00000000, 0.00000000,1.00000000,1.00000000,68493.1507, 1.00000000,0.00000000,0.00000000,0.00000000, 0.00000000,1.00000000,0.00000000,0.00000000, 0.00000000,0.00000000,0.00000000,0.00000000, 0.00000000,0.00000000,0.00000000,0.00000000, 0.00000000,0.00000000,0.00000000,0.00000000, 1.00000000,0.00000000,0.00000000,0.00000000, 0.00000000,0.00000000,0.00000000,0.00000000, 0.00000000,0.00000000,0.00000000,0.00000000, 0.00000000,0.00000000,0.00000000,1.00000000, 0.00000000,0.00000000,0.00000000,0.00000000, 0.00000000,0.00000000,0.00000000,0.00000000, 1.00000000,0.00000000,0.00000000,68493.1507, 68493.1507,1369.86301,1.00000000,0.00000000, 0.00000000,0.00000000,0.00000000,0.00000000, 0.00000000,1.00000000,5479.45205,0.00000000, 0.00000000,0.00000000,0.00000000,0.00000000, 0.00000000,0.00000000,1.00000000,0.00000000, 0.00000000,68493.1507,1.00000000,68493.1507, 68493.1507,68493.1507,0.00000000,0.00000000, 0.00000000,0.00000000,0.00000000,0.00000000, 0.00000000,0.00000000,0.00000000,0.00000000, 68493.1507,0.00000000,0.00000000,1369.86301, 0.00000000,0.00000000,1.00000000,0.00000000, 0.00000000,0.00000000,0.00000000,0.00000000, 0.00000000,0.00000000,0.00000000,0.00000000, 0.00000000,68493.1507,0.00000000,0.00000000, 0.00000000,0.00000000,1.00000000,0.00000000, 0.00000000,1.00000000,0.00000000,0.00000000, 0.00000000,0.00000000,0.00000000,1.00000000, 0.00000000,0.00000000,0.00000000,0.00000000, 1.00000000,0.00000000,1.00000000,0.00000000, 0.00000000,0.00000000,0.00000000,0.00000000, 0.00000000,0.00000000,1.00000000,0.00000000, 0.00000000,0.00000000,1.00000000,0.00000000, 0.00000000,0.00000000,1.00000000,1.00000000, 0.00000000,0.00000000,0.00000000,0.00000000, 0.00000000,0.00000000,0.00000000,0.00000000, 0.00000000,0.00000000,1.00000000,68493.1507, 1.00000000,0.00000000,0.00000000,1369.86301, 0.00000000,0.00000000,0.00000000,0.00000000, 0.00000000,1.00000000,0.00000000,1.00000000, 0.00000000,0.00000000,0.00000000,68493.1507, 1.00000000,0.00000000,0.00000000,1.00000000, 0.00000000,0.00000000,0.00000000,0.00000000, 0.00000000,0.00000000,0.00000000,1.00000000, 0.00000000,0.00000000,0.00000000,0.00000000, 0.00000000,0.00000000,1.00000000,0.00000000, 0.00000000,0.00000000,0.00000000,0.00000000, 0.00000000,1.00000000,2054.79452,0.00000000, 0.00000000,0.00000000,1.00000000,0.00000000, 0.00000000,0.00000000,0.00000000,1.00000000, 0.00000000,0.00000000,1.00000000,0.00000000, 1369.86301,0.00000000,0.00000000,0.00000000, 0.00000000,0.00000000,1.00000000,0.00000000, 0.00000000,68493.1507,0.00000000,0.00000000, 0.00000000,0.00000000,0.00000000,0.00000000, 0.00000000,0.00000000,1.00000000,0.00000000, 0.00000000,0.00000000,0.00000000,1.00000000, 0.00000000
        }, 1, 289);

        File file = new File(testDir.newFolder(), "file.json");
        BinarySerde.writeArrayToDisk(arr, file);
        requestSpecification.body(jsonObject.encode().getBytes());
        requestSpecification.header("Content-Type", "multipart/form-data");
        String response = requestSpecification.when()
                .multiPart("default", file)
                .expect().statusCode(200)
                .body(not(isEmptyOrNullString()))
                .post("/regression/nd4j").then()
                .extract()
                .body().asString();

        File outputImagePath = new File(testDir.newFolder(), "file.json");
        FileUtils.writeStringToFile(outputImagePath, response, Charset.defaultCharset());
        INDArray outputArray = BinarySerde.readFromDisk(outputImagePath);
        INDArray expectedArr = ExpectedAssertUtil.fileAndKeyToNDArrayOutput("src/test/resources/Json/scikitlearn/ScikitlearnNdArrayTest.json", "raw");
        Assert.assertEquals(expectedArr.getInt(0), outputArray.getInt(0));
        Assert.assertEquals(expectedArr, outputArray);

    }

}
