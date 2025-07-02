# terraform/s3.tf
resource "aws_s3_bucket" "model_artifacts" {
  bucket = "gacp-model-artifacts-${random_id.bucket_suffix.hex}"

  tags = {
    Name        = "GACP Model Artifacts"
    Environment = "production"
  }
}

resource "aws_s3_bucket" "ab_testing_data" {
  bucket = "gacp-ab-testing-data-${random_id.bucket_suffix.hex}"

  tags = {
    Name        = "GACP AB Testing Data"
    Environment = "production"
  }
}

resource "aws_s3_bucket" "farmer_images" {
  bucket = "gacp-farmer-images-${random_id.bucket_suffix.hex}"

  tags = {
    Name        = "GACP Farmer Images"
    Environment = "production"
  }
}

resource "aws_s3_bucket_versioning" "model_artifacts_versioning" {
  bucket = aws_s3_bucket.model_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "ab_testing_lifecycle" {
  bucket = aws_s3_bucket.ab_testing_data.id

  rule {
    id     = "ab_testing_retention"
    status = "Enabled"

    expiration {
      days = 90
    }

    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }
}

resource "random_id" "bucket_suffix" {
  byte_length = 8
}