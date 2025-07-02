# terraform/dynamodb.tf
resource "aws_dynamodb_table" "ab_testing_results" {
  name           = "gacp-ab-testing-results"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "experiment_id"
  range_key      = "timestamp"

  attribute {
    name = "experiment_id"
    type = "S"
  }

  attribute {
    name = "timestamp"
    type = "N"
  }

  attribute {
    name = "user_id"
    type = "S"
  }

  global_secondary_index {
    name            = "user-timestamp-index"
    hash_key        = "user_id"
    range_key       = "timestamp"
    projection_type = "ALL"
  }

  tags = {
    Name        = "GACP AB Testing Results"
    Environment = "production"
  }
}

resource "aws_dynamodb_table" "model_performance" {
  name           = "gacp-model-performance"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "model_id"
  range_key      = "version_timestamp"

  attribute {
    name = "model_id"
    type = "S"
  }

  attribute {
    name = "version_timestamp"
    type = "N"
  }

  tags = {
    Name        = "GACP Model Performance"
    Environment = "production"
  }
}

resource "aws_dynamodb_table" "farmer_analytics" {
  name           = "gacp-farmer-analytics"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "farmer_id"
  range_key      = "metric_date"

  attribute {
    name = "farmer_id"
    type = "S"
  }

  attribute {
    name = "metric_date"
    type = "S"
  }

  tags = {
    Name        = "GACP Farmer Analytics"
    Environment = "production"
  }
}

resource "aws_dynamodb_table" "prediction_audit" {
  name           = "gacp-prediction-audit"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "prediction_id"
  range_key      = "timestamp"

  attribute {
    name = "prediction_id"
    type = "S"
  }

  attribute {
    name = "timestamp"
    type = "N"
  }

  stream_enabled   = true
  stream_view_type = "NEW_AND_OLD_IMAGES"

  tags = {
    Name        = "GACP Prediction Audit"
    Environment = "production"
  }
}