package com.databricks.mlflow.mleap;

import com.databricks.mlflow.mleap.MLeapPredictor;
import com.databricks.mlflow.models.Predictor;
import com.databricks.mlflow.LoaderModule;

import java.util.Optional;
import java.io.IOException;

public class MLeapLoader extends LoaderModule<MLeapFlavor> {
    @Override
    protected Predictor createPredictor(String modelDataPath) {
        return new MLeapPredictor(modelDataPath);
    }

    @Override
    protected Class<MLeapFlavor> getFlavorClass() {
        return MLeapFlavor.class;
    }

    @Override
    protected String getFlavorName() {
        return MLeapFlavor.FLAVOR_NAME;
    }
}
